import sys
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm # Use tqdm.notebook for cleaner outputs in notebooks.
 
# Also import local helpers
target_dir = os.path.abspath('..') 

if target_dir not in sys.path:
    sys.path.append(target_dir)

# have GPU available to speed up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
from football_lstm import FootballLSTM
from preprocess import merge_stats_df_with_transfermarkt

# hypertuning of LSTM 
def hyperparam_tuning(params: dict, stats_df: pd.DataFrame, train_dataloader: torch.utils.data.DataLoader, 
                          test_dataloader: torch.utils.data.DataLoader):
        
    best_combo = None
    previous_best = None

    for lr in tqdm(params['learning_rates'], desc="learning_rates", position=0):
        for epoch in tqdm(params['epochs'], desc="epochs", position=1, leave=False):
            for layer in tqdm(params['layers'], desc="layers", position=2, leave=False):
                for h_size in tqdm(params['h_sizes'], desc="h_sizes", position=4, leave=False):
                    for dropout in tqdm(params['dropouts'], desc="dropouts", position=5, leave=False):

                        # non-zero dropout expects num_layers greater than 1 so train with dropout set to 0
                        if layer == 1:
                            model = FootballLSTM(n_features=len(stats_df.columns), hidden_size=h_size, num_layers=layer, dropout=0).to(device)
                        else:
                            model = FootballLSTM(n_features=len(stats_df.columns), hidden_size=h_size, num_layers=layer, dropout=dropout).to(device)
                                
                        loss_fn = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                            
                        # Train model
                        train_losses, test_losses = model.train_model(
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            n_epochs=epoch
                        )

                        # model evaluation
                        rmse = model.eval_model(test_dataloader)["Overall"][0]

                        # evaluating if better setup, if so save
                        if previous_best is None or rmse < previous_best:
                            # manually set dropout to 0 if layer is 1
                            best_combo = [lr, epoch, layer, h_size, 0 if layer == 1 else dropout]
                            previous_best = rmse
                            best_train_losses, best_test_losses = train_losses, test_losses

    # save to dictionary for more clear access
    param_dict = {}
    param_dict['learning_rate'] = best_combo[0]
    param_dict['epoch'] = best_combo[1]
    param_dict['layers'] = best_combo[2]
    param_dict['hidden_size'] = best_combo[3]
    param_dict['dropout'] = best_combo[4]

    plt.plot(best_train_losses, color="red", label="Train loss")
    plt.plot(best_test_losses, color="blue", label="Test loss")
    plt.legend()
    plt.title(f"""Hyperparameter Tuning - Train vs Test Loss (lr: {param_dict['learning_rate']}, epoch: {param_dict['epoch']}, 
            layers: {param_dict['layers']}, hidden size: {param_dict['hidden_size']}, dropout: {param_dict['dropout']})""")
    # ran 100min hyperparameter tuning, and dir was not there as I had not saved any graphs yet, so make dir if it does not exist
    os.makedirs("tuning_graphs", exist_ok=True)
    plt.savefig(f"tuning_graphs/"f"lr{param_dict['learning_rate']}_ep{param_dict['epoch']}_ly{param_dict['layers']}_hs{param_dict['hidden_size']}_do{param_dict['dropout']}.png")
    plt.close()
        
    return param_dict

def get_actuals_vs_predictions_df(stats_df: pd.DataFrame, model: nn.Module, blocks_per_input: int, max_look_ahead: int = None):
    """
    For the given stats_df, uses the given model to make predictions for each
    player, before joining that prediction data with real Transfermarkt values.
    This provides actual/prediction pairs to help evaluate the Bayesian model.
    
    max_look_ahead is uses to control how many blocks ahead to predict per player.
    If none, predicts as many blocks as the player has.
    """
    if max_look_ahead is None:
        max_look_ahead = float('inf')
    
    # can't assume we know the dates so predict them too
    def get_k_future_dates(player_df, k):
        # Get the dates of existing blocks
        block_dates = player_df.index.get_level_values('date')
    
        # Calculate average time between consecutive blocks
        deltas = pd.Series(block_dates).diff().dropna()
        avg_delta = deltas.mean()
    
        # Project forward from last known date
        last_date = block_dates[-1]
        future_dates = [last_date + avg_delta * i for i in range(1, k + 1)]
    
        return future_dates
    
    ids = []
    names = []
    dates = []
    leagues = []
    preds = []
    
    # Over each player
    for (id, name), player_df in stats_df.groupby(["player_id", "player_name"]):
        vals = player_df.values

        if len(vals) <= blocks_per_input:
            continue
        
        x = torch.tensor(vals[:blocks_per_input], dtype=torch.float32).unsqueeze(0)
        remaining_games = min(len(vals) - blocks_per_input, max_look_ahead)
        player_preds = model.predict_next_k(x, k=remaining_games).squeeze(0).detach().cpu().numpy()
        
        preds.append(player_preds)
        ids.extend([id] * remaining_games)
        names.extend([name] * remaining_games)
        dates.extend(get_k_future_dates(player_df.iloc[:blocks_per_input], remaining_games))
        leagues.extend([player_df.index.get_level_values('league')[blocks_per_input]] * remaining_games) 
        
    preds = np.concatenate(preds)
    
    # Turn into a df
    preds_df = pd.DataFrame(preds)
    # Add id and name columns
    preds_df.insert(0, 'player_id', ids)
    preds_df.insert(1, 'player_name', names)
    preds_df.insert(2, 'date', dates)
    preds_df.insert(3, 'league', leagues)

    # Rename other columns
    stats_cols = stats_df.columns
    
    preds_df = preds_df.rename(columns={i: stats_cols[i] for i in range(len(stats_cols))})
    
    #print(preds_df)
    
    # Pull transfer value data and merge
    preds_df_combined = merge_stats_df_with_transfermarkt(preds_df, False)
    
    # Add age and year (fractional columns
    preds_df_combined["age"] = (preds_df_combined["date"] - preds_df_combined["date_of_birth"]).dt.days  / 365.25 # .25 accounts for leap years

    preds_df_combined["year"] = preds_df_combined["date"].dt.year + (preds_df_combined["date"].dt.day_of_year / 365.25)
    
    # Get rid of DOB column
    preds_df_combined = preds_df_combined.drop("date_of_birth", axis=1)
    
    return preds_df_combined