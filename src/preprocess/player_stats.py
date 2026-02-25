from typing import List, Set

import pandas as pd
from understatapi import UnderstatClient

"""
This module contains basic helpers to load in player stats using understat.
"""

def get_player_ids(understat: UnderstatClient, positions: Set[str], league: str = "EPL", season: str = "2025") -> List[str]:
    """
    Gets a list of all player ids for the given league, season, and positions.
    """
    players = understat.league(league=league).get_player_data(season=season)
    
    # Filter by position
    pos_players = list(filter(lambda p: any(pp in positions for pp in p["position"].split(" ")), players))
    return [p["id"] for p in pos_players]
    
def get_player_stats_df(understat: UnderstatClient, player_id: str,
                      stats: List[str]) -> pd.DataFrame:
    """
    Produces a dataframe with per-90 stats for every game played by the given
    player_id, for any club or season played.
    """
    # Get all player matches
    player_matches = understat.player(player=player_id).get_match_data()
        
    # Convert to dataframe
    player_matches_df = pd.DataFrame(player_matches)
        
    # Sort by date
    player_matches_df = player_matches_df.sort_values(by="date")
        
    # Per-90 stats over games
    per_90_stats = list(map(lambda s: f"{s}_per_90", stats))

    # Stats/90 for each
    for stat, per_90_stat in zip(stats, per_90_stats):
        player_matches_df[per_90_stat] = player_matches_df[stat] / player_matches_df["time"] * 90

    # Only date plus stats
    stats_df = player_matches_df[["date"] + per_90_stat]
        
    return stats_df