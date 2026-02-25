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
    players = understat.league(league="EPL").get_player_data(season="2025")
    
    # Filter by position
    pos_players = list(filter(lambda p: any(pp in positions for pp in p["position"].split(" ")), players))
    return [p["id"] for p in pos_players]
    
def get_player_stats_df(understat: UnderstatClient, player_id: str,
                      stats: List[str], window_size: int=10) -> pd.DataFrame:
    """
    Produces a dataframe with rolling per-90 stats for each in the given list of stats,
    using the given player_id and window size.
    """
    # Get all player matches
    player_matches = understat.player(player=player_id).get_match_data()
        
    # Convert to dataframe
    player_matches_df = pd.DataFrame(player_matches)
        
    # Sort by date
    player_matches_df = player_matches_df.sort_values(by="date")
        
    # Rolling stats over games
    rolling_stats = list(map(lambda s: f"rolling_{s}_per_90", stats))

    # Get minute counts first
    player_matches_df["rolling_min"] = player_matches_df["time"].rolling(window_size).sum()

    # Rolling stats/90 for each
    for stat, rolling_stat_name in zip(stats, rolling_stats):
        rolling_stat = player_matches_df[stat].rolling(window_size).sum()
        player_matches_df[rolling_stat_name] = rolling_stat / player_matches_df["rolling_min"] * 90

    # Only date plus stats
    stats_df = player_matches_df[["date"] + rolling_stats]
        
    # Drop nas (first window_size - 1)
    stats_df = stats_df.dropna()
        
    return stats_df

