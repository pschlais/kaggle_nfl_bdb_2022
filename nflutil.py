import pandas as pd
import numpy as np
import os.path


# ---------- INTERNAL FUNCTIONS ----------------- #####
def _rgb(r, g, b):
    # Converts RGB range from 0-255 to 0-1 for matplotlib.
    # Returns np.array of length 3.
    return np.array([r, g, b]) / 255.0


# --------- CONSTANTS --------------------------- ####
FIELD_SIZE_X = 120.0  # yards, field goal to field goal (back of endzones)
FIELD_SIZE_Y = 53.3  # yards, sideline to sideline
TEAM_COLORS = {'ARI': {'main': _rgb(155, 35, 63), 'secondary': _rgb(255, 255, 255)},
               'ATL': {'main': _rgb(0, 0, 0), 'secondary': _rgb(255, 255, 255)},
               'BAL': {'main': _rgb(26, 25, 95), 'secondary': _rgb(255, 255, 255)},
               'BUF': {'main': _rgb(0, 51, 141), 'secondary': _rgb(198, 12, 48)},
               'CAR': {'main': _rgb(0, 133, 202), 'secondary': _rgb(16, 24, 32)},
               'CHI': {'main': _rgb(11, 22, 42), 'secondary': _rgb(200, 56, 3)},
               'CIN': {'main': _rgb(251, 79, 20), 'secondary': _rgb(255, 255, 255)},
               'CLE': {'main': _rgb(255, 60, 0), 'secondary': _rgb(49, 29, 0)},
               'DAL': {'main': _rgb(0, 34, 68), 'secondary': _rgb(134, 147, 151)},
               'DEN': {'main': _rgb(0, 34, 68), 'secondary': _rgb(251, 79, 20)},
               'DET': {'main': _rgb(0, 118, 182), 'secondary': _rgb(176, 183, 188)},
               'GB': {'main': _rgb(24, 48, 40), 'secondary': _rgb(255, 184, 28)},
               'HOU': {'main': _rgb(3, 32, 47), 'secondary': _rgb(167, 25, 48)},
               'IND': {'main': _rgb(0, 44, 95), 'secondary': _rgb(162, 170, 173)},
               'JAX': {'main': _rgb(0, 103, 120), 'secondary': _rgb(16, 24, 32)},
               'KC': {'main': _rgb(227, 24, 55), 'secondary': _rgb(255, 255, 255)},
               'LA': {'main': _rgb(0, 53, 148), 'secondary': _rgb(255, 209, 0)},
               'LAC': {'main': _rgb(0, 42, 94), 'secondary': _rgb(255, 194, 14)},
               'MIA': {'main': _rgb(0, 142, 151), 'secondary': _rgb(252, 76, 2)},
               'MIN': {'main': _rgb(79, 38, 131), 'secondary': _rgb(255, 255, 255)},
               'NE': {'main': _rgb(0, 34, 68), 'secondary': _rgb(176, 183, 188)},
               'NO': {'main': _rgb(211, 188, 141), 'secondary': _rgb(16, 24, 31)},
               'NYG': {'main': _rgb(1, 35, 82), 'secondary': _rgb(163, 13, 45)},
               'NYJ': {'main': _rgb(18, 87, 64), 'secondary': _rgb(255, 255, 255)},
               'OAK': {'main': _rgb(0, 0, 0), 'secondary': _rgb(165, 172, 175)},
               'PHI': {'main': _rgb(0, 76, 84), 'secondary': _rgb(165, 172, 175)},
               'PIT': {'main': _rgb(16, 24, 32), 'secondary': _rgb(255, 182, 18)},
               'SEA': {'main': _rgb(0, 34, 68), 'secondary': _rgb(105, 190, 40)},
               'SF': {'main': _rgb(170, 0, 0), 'secondary': _rgb(173, 153, 93)},
               'TB': {'main': _rgb(213, 10, 10), 'secondary': _rgb(10, 10, 8)},
               'TEN': {'main': _rgb(12, 35, 64), 'secondary': _rgb(75, 146, 219)},
               'WAS': {'main': _rgb(63, 16, 16), 'secondary': _rgb(255, 182, 18)}}


# -------------- FUNCTIONS -------------------------------------------
def base_import(base_path='csv', week=1):
    """
    Imports DataFrames of data
    :param base_path: [str] folder path containing csv files
    :param week: [int] week number of tracking data to import (1-17)
    :return: tuple of dataframes (games, plays, players, coverages, targeted_receiver, track_data)
    """
    track_file_name = 'week' + str(week) + '.csv'
    game_df = pd.read_csv(os.path.join(base_path, 'games.csv'))
    play_df = pd.read_csv(os.path.join(base_path, 'plays.csv'))
    players_df = pd.read_csv(os.path.join(base_path, 'players.csv'))
    coverage_df = pd.read_csv(os.path.join(base_path, 'coverages_week1.csv'))
    target_df = pd.read_csv(os.path.join(base_path, 'targetedReceiver.csv'))
    track_df = pd.read_csv(os.path.join(base_path, track_file_name))

    return game_df, play_df, players_df, coverage_df, target_df, track_df


def transform_tracking_data(track_df, inplace=False):
    """
    Standardizes the tracking data so that all offensive plays have the same reference frame (Madden camera)

    Transforms all positional attributes: x, y, o, dir. Returns a copy of the dataframe. Increasing x is downfield for
    the offense, increasing y is towards the left sideline
    :param track_df: [DataFrame] tracking data
    :param inplace: [boolean] determine if transformation should be made in place or on a copy of the DataFrame
    :return: DataFrame if inplace=False, otherwise None
    """
    if inplace:
        out_df = track_df  # work on actual DataFrame
    else:
        out_df = track_df.copy()  # only apply changes to a copy of the DataFrame

    # get indexing for plays moving to the left (right is the standard)
    i_left = (out_df.playDirection == 'left')

    # transform X, Y
    out_df.loc[i_left, 'x'] = FIELD_SIZE_X - out_df.loc[i_left, 'x']
    out_df.loc[i_left, 'y'] = FIELD_SIZE_Y - out_df.loc[i_left, 'y']

    # convert the orientations so that 0 points to left sideline
    # - see figure here for more info (https://www.kaggle.com/c/nfl-big-data-bowl-2021/data)
    out_df.loc[i_left, 'o'] = (out_df.loc[i_left, 'o'] + 180) % 360
    out_df.loc[i_left, 'dir'] = (out_df.loc[i_left, 'dir'] + 180) % 360

    if inplace:
        return None  # modified inplace
    else:
        return out_df
