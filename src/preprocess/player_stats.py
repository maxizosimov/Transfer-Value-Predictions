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
    
def get_player_stats_df(understat: UnderstatClient,
                        player_id: str,
                        games_per_block: int,
                        stats: List[str]) -> pd.DataFrame:
    """
    Produces a dataframe with aggregate per-90 stats for every games_per_block games played by the given
    player_id, for the given leagues and seasons. Indexes by date.
    
    Note each row contains data for a disjoint set of games.
    
    Note if the player played less than games_per_block games, an empty dataframe
    will be returned.
    """
    # Turn into sets for easy membership checks
    leagues = set(leagues)
    seasons = set(seasons)
    
    # Get all player matches
    player_matches = understat.player(player=player_id).get_match_data()
        
    # Convert to dataframe
    player_matches_df = pd.DataFrame(player_matches)
        
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
            
        return row
        
    # Construct aggregated_df
    rows = []
    
    for i in range(0, len(player_matches_df) - games_per_block, games_per_block):
        rows.append(aggregate_in_window(player_matches_df.iloc[i:i + games_per_block]))
        
    agg_df = pd.DataFrame(rows)
    
    #print(agg_df.head())

    # Set index to date
    if not agg_df.empty: # Below will error for empty df
        agg_df = agg_df.set_index("date")
        
    return agg_df

def get_player_stats_df_from_info(games_per_block: int,
                                  player_info: Dict,
                                  stats: List[str]):
    """
    Produces player per-90 stats but on the given info dict, not using understat
    at all.
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
    with aggregate per-90 stats, for every block of games_per_block games played by each of the given
    player_ids, for any club or season played. Indexes by player_id and date.
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
                
                except Exception as e: # 
                    print(f"Caught error {e}, skipping match.")
                    skipped_matches += 1
    
    print(f"Skipped {skipped_matches} matches total.")
      
    # 2. Process players from disk and incrementally write to parquet (faster than csv)             
    for filename in os.listdir(dir_name):
        player_info = []
        with open(f"{dir_name}/{filename}", 'r') as f:
            for line in f:
                player_info.append(json.loads(line))
    
        stats_df = get_player_stats_df_from_info(games_per_block, player_info, stats)
    
        if not stats_df.empty:
            player_id = filename.replace('.jsonl', '')
            stats_df.to_parquet(f"{dir_name}/{player_id}.parquet")
    
        del player_info
        del stats_df

    # Combine all parquet files at the end
    result = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{dir_name}/*.parquet")])

    shutil.rmtree(dir_name)
    return result

class CustomFootballDataset(Dataset):
    """
    A torch dataset to wrap around a stats dataframe for training a time series
    model.
    
    At a given index, one can get a pairing containing metrics for the last 
    blocks_per_input game blocks, along with the current game's metrics as the label.
    """
    
    def __init__(self, stats_df: pd.DataFrame, blocks_per_input: int = 10, multiple_players: bool = True):
        """
        Initializes a CustomFootballDataset over the given stats_df, storing model
        inputs and outputs in X and y respectively. Each value in X provides
        a 2d array of stats for the last blocks_per_input game blocks, while each value
        in y provides stats for the current game to predict with the matching
        X values.
        """
        super().__init__()
        
        self.X = []
        self.y = []
        
        if multiple_players:   
            # Break down by player
            for _, player_df in stats_df.groupby("player_id"):
                vals = player_df.values
                for i in range(len(vals) - blocks_per_input):
                    self.X.append(vals[i:i + blocks_per_input])
                    self.y.append(vals[i + blocks_per_input])
                    
        else:
            vals = stats_df.values
            for i in range(len(vals) - blocks_per_input):
                self.X.append(vals[i:i + blocks_per_input])
                self.y.append(vals[i + blocks_per_input])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)