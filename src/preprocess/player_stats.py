from typing import List, Set, Dict
from collections import defaultdict
import os
import json
import shutil
import glob

import pandas as pd
from understatapi import UnderstatClient
import torch
from torch.utils.data import Dataset
import duckdb
import numpy as np

"""
This module contains basic helper functions or classes to load in or store
player stats using understat.
"""

def get_player_ids(understat: UnderstatClient, positions: List[str], league: str = "EPL", season: str = "2025") -> List[str]:
    """
    Gets a list of all player ids for the given positions in the given league and season.
    """
    positions = set(positions)
    
    players = understat.league(league=league).get_player_data(season=season)
    
    # Filter by position
    pos_players = list(filter(lambda p: any(pp in positions for pp in p["position"].split(" ")), players))
    return [p["id"] for p in pos_players]

def get_player_stats_df_from_info(games_per_block: int,
                                  player_info: Dict,
                                  stats: List[str]):
    """
    For the given dict of player info, produces a pandas dataframe for the
    given stats, which are aggregated into windows of size games_per_block.
    
    DF includes additional info including player_id, player_name, date, and
    league.
    """
    #print(player_info)
    
    # Convert to dataframe
    player_matches_df = pd.DataFrame(player_info)
    
    # Sort by date
    player_matches_df = player_matches_df.sort_values(by="date")
    
    # Per-90 stats over games
    per_90_stats = list(map(lambda s: f"{s}_per_90", stats))
    
    def aggregate_in_window(window_df):
        """Constructs a particular aggregated row with per-90 stats."""
        mins = window_df["time"].astype(int).sum()
        
        row = {}
        
        for stat, per_90_stat in zip(stats, per_90_stats):
            row[per_90_stat] = window_df[stat].astype(float).sum() / mins * 90
        
        # For date, take the max
        row["date"] = max(window_df["date"])
        
        # For league, take last
        row["league"] = window_df["league"].iloc[-1] # Get last
            
        return row
        
    # Construct aggregated_df
    rows = []
    
    for i in range(0, len(player_matches_df) - games_per_block, games_per_block):
        rows.append(aggregate_in_window(player_matches_df.iloc[i:i + games_per_block]))
        
    agg_df = pd.DataFrame(rows)
    
    # Add player id col
    agg_df["player_id"] = player_matches_df["player_id"][0]
    
    # Add player name col
    agg_df["player_name"] = player_matches_df["player"][0]
    
    #print(agg_df.head())

    # Set index to date and name and  player_id
    if not agg_df.empty: # Below will error for empty df
        agg_df = agg_df.set_index(["player_id", "player_name", "date"])
        
    return agg_df

def get_position_players_stats_df(understat: UnderstatClient, positions: List[str],
                                  games_per_block: int,
                                  stats: List[str],
                                  leagues: List[str] = ["EPL"],
                                  seasons: List[str] = [2025]) -> pd.DataFrame:
    """
    Produces a dataframe of all players for the given position, leagues, and seasons
    with aggregate per-90 stats, for every block of games_per_block games played.
    Indexes by player_id, player_name and date.
    """
    
    player_info_map = defaultdict(list)
    
    positions = set(positions)
    
    # 1. Get mapping of players to their game data, for the relevant leagues, save to disk to fix memory issues.
    
    dir_name = "../data/tmp_player_data"
    
    os.makedirs(dir_name, exist_ok=True)
    
    skipped_matches = 0 # Count how many skipped
    
    for league in leagues:
        for season in seasons:
            # Get ids for given position in league and season
            player_ids = set(get_player_ids(understat, list(positions), league, season))
            
            # Get all matches in this league/season
            matches = understat.league(league=league).get_match_data(season=str(season))
        
            for match in matches:
                
                try:
                
                    match_id = match['id']
                    # Get player stats for this specific match
                
                    roster = understat.match(match=match_id).get_roster_data()
                
                    date = match['datetime']
            
                    for side in ['h', 'a']:
                        for _, player_info in roster[side].items():
                            if player_info['player_id'] in player_ids:
                                player_info['date'] = date
                                player_info['league'] = league
                        
                                # Append to player's file
                                filepath = f"{dir_name}/{player_info['player_id']}.jsonl"
                                with open(filepath, 'a') as f:
                                    f.write(json.dumps(player_info) + '\n')
                
                except Exception as e: # Understat seems to have a number of invalid matches, each with their own errors.
                    print(f"Caught error {e}, skipping match.")
                    skipped_matches += 1
    
    print(f"Skipped {skipped_matches} matches total.")
      
    # 2. Process players from disk and incrementally write to parquet (faster than csv)             
    for filename in os.listdir(dir_name):
        player_info = []
        with open(f"{dir_name}/{filename}", 'r') as f:
            for line in f:
                player_info.append(json.loads(line))
    
        # Get into windows
        stats_df = get_player_stats_df_from_info(games_per_block, player_info, stats)
    
        if not stats_df.empty:
            player_id = filename.replace('.jsonl', '')
            stats_df.to_parquet(f"{dir_name}/{player_id}.parquet")
    
        # freee up memory
        del player_info
        del stats_df

    # Combine all parquet files at the end
    result = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{dir_name}/*.parquet")])

    # get rid of temp dir
    shutil.rmtree(dir_name)
    return result

class CustomFootballDataset(Dataset):
    """
    A torch dataset to wrap around a stats dataframe for training a time series
    model.
    
    At a given index, one can get a pairing containing metrics for the last 
    blocks_per_input game blocks, along with the current game's metrics as the label,
    and the player id.
    """
    
    def __init__(self, stats_df: pd.DataFrame, blocks_per_input: int = 10, multiple_players: bool = True):
        """
        Initializes a CustomFootballDataset over the given stats_df, storing model
        inputs, outputs, and player_id ahead for the player in X, y, and
        player_ids respectively. Each value in X provides
        a 2d array of stats for the last blocks_per_input game blocks, each value
        in y provides stats for the current game to predict with the matching
        X values.
        """
        super().__init__()
        
        self.X = []
        self.y = []
        self.player_ids = []
        
        if multiple_players:   
            # Break down by player
            for id, player_df in stats_df.groupby("player_id"):
                #print(player_df)
                vals = player_df.values
                for i in range(len(vals) - blocks_per_input):
                    self.X.append(vals[i:i + blocks_per_input])
                    self.y.append(vals[i + blocks_per_input])
                    self.player_ids.append(id)
                    
        else:
            vals = stats_df.values
            for i in range(len(vals) - blocks_per_input):
                self.X.append(vals[i:i + blocks_per_input])
                self.y.append(vals[i + blocks_per_input])
                self.player_ids.append(stats_df["player_id"].iloc[0]) # Assume its always the same player

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), self.player_ids[idx]

def merge_stats_df_with_transfermarkt(stats_df: pd.DataFrame, use_transfermarkt_info=True) -> pd.DataFrame:
    """
    For the given stats_df, merges it with pulled transfermarkt data to include
    additional features like DOB and real market value.
    
    use_transfermarkt_info controls whether to use Transfermarkt's league and
    date info, or what is in stats_df.
    """
    # Change date dtype for consistency in merge_asof func
    stats_df["date"] = stats_df['date'].astype('datetime64[us]')
    
    # Sort by date
    stats_df = stats_df.sort_values(by="date")
    
    # 2. Pull transfer value data
    con = duckdb.connect()

    q = """
    SELECT name player_name, date_of_birth, V.date AS t_date, V.market_value_in_eur AS value, V.player_club_domestic_competition_id league
    FROM read_csv_auto('https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz') P
    JOIN read_csv_auto('https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/player_valuations.csv.gz') V ON P.player_id = V.player_id
    AND V.player_club_domestic_competition_id IN ('GB1', 'ES1', 'GR1', 'FR1', 'IT1')
    AND YEAR(V.date) >= 2014
    ORDER BY V.date ASC
    """

    transfer_df = con.sql(q).df()
    
    if use_transfermarkt_info:
        stats_df = stats_df.drop("league",axis=1)
        # Map to Understat's league names for consistency
        transfer_df["league"] = transfer_df["league"].map({
            "GB1":"EPL",
            "ES1":"La_Liga",
            "GR1":"Bundesliga",
            "FR1":"Ligue_1",
            "IT1":"Serie_A"
        })
    else:
        transfer_df = transfer_df.drop("league",axis=1)
    
    # Already ordered by date
    transfer_df["t_date"] = transfer_df['t_date'].astype('datetime64[us]')
    transfer_df["date_of_birth"] = transfer_df['date_of_birth'].astype('datetime64[us]')
    
    # 3. Join transfer data with performance data - join on latest performance date that's no later than the value date
    stats_df_combined = pd.merge_asof(transfer_df, stats_df, left_on="t_date", right_on="date", by="player_name")
    
    # Determine which date to keep
    if use_transfermarkt_info:
        stats_df_combined["date"] = stats_df_combined["t_date"]
    # Otherwise just keep the date from stats_df
    
    # Drop those with nas
    stats_df_combined = stats_df_combined.dropna()
    
    #per_90
    stats_cols = list(filter(lambda s: s[-6:]=="per_90", stats_df.columns))
    
    # Rearrange columns
    stats_df_combined = stats_df_combined[["player_id","player_name","date","date_of_birth","league"] + stats_cols + ["value"]]
    
    return stats_df_combined