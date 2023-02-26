import pandas as pd
import numpy as np
import nflutil

# ===== PREP FUNCTIONS ========================= #
def prep_get_modeling_frames(track_df, play_df, pff_df):
    return (
        track_df
        # (1) filter down to punt plays with a return
        .merge(play_df.loc[(play_df.specialTeamsPlayType=='Punt') & (play_df.specialTeamsResult=='Return'), ['gameId','playId']],
               how='inner',
               on=['gameId','playId']
              )
        # (2) filter down to clean catches
        .merge(pff_df.loc[pff_df.kickContactType=='CC', ['gameId','playId']],
               how='inner',
               on=['gameId','playId']
              )
        # (3) filter to all frames between the punt and the catch
        .merge(
            # Dataframe of: gameId, playId, punt (frameId), punt_received (frameId)
            (nflutil.get_frame_of_event(track_df, 
                                   ['punt','punt_received'])
             .pivot(index=['gameId','playId'], columns='event', values='frameId')
             .dropna()  # remove any rows where punt OR punt_received events are not present (both means a valid play)
             .rename(columns={'punt': 'puntFrameId', 'punt_received': 'puntReceivedFrameId'})
             .astype('int64')
             .reset_index()
                ),
            how='left',
            on=['gameId','playId']
            )
        .query('(frameId >= puntFrameId) & (frameId <= puntReceivedFrameId)')

        # delete intermediate columns
        .drop(columns=['puntFrameId', 'puntReceivedFrameId'])
        .reset_index(drop=True)
        )


# ==== FEATURE GENERATION FUNCTIONS =========================================
def feat_timeToCatch(track_df):
    # adds timeToCatch feature to dataframe
    return (
        track_df
        .merge(
            # Dataframe of: gameId, playId, punt_received (frameId)
            (nflutil.get_frame_of_event(track_df, 'punt_received')
             .pivot(index=['gameId','playId'], columns='event', values='frameId')
             .rename(columns={'punt_received': 'puntReceivedFrameId'})
             .astype('int64')
             .reset_index()
                ),
            how='left',
            on=['gameId','playId']
        )
        .assign(timeToCatch=lambda df_: (df_['puntReceivedFrameId'] - df_['frameId'])/10)
        .drop(columns=['puntReceivedFrameId'])
    )


def feat_byDefender(track_df, play_df, game_df, n_defenders=4):
    return_df =  (
        track_df
        # get location data of returner for each frame for modeling
        .merge(play_df.loc[~play_df.returnerId.astype(str).str.contains(';'), ## only consider plays with 1 returner
                            ['gameId','playId','returnerId']].astype({"returnerId": float}),
                how='inner',
                left_on=['gameId','playId','nflId'],
                right_on=['gameId','playId','returnerId'])
        .filter(['gameId','playId','frameId','team','x','y'])
        .rename(columns={'x':'x_returner', 'y':'y_returner', 'team':'teamReturner'})
        # attach the returner data to the rest of the tracking data
        .merge(track_df[['gameId','playId','frameId','nflId','team','x','y','s','o','dir']],
                how='left',
                on=['gameId','playId','frameId'])
        # attach home and away team labels for each game
        .merge(game_df[['gameId','homeTeamAbbr','visitorTeamAbbr']],
                how='inner',
                on='gameId')
        # filter to opposing team players
        .query('teamReturner != team and team != "football"')
        # assemble identifier for punting team players
        .assign(puntTeamAbbr=lambda df_: np.where(df_['team']=='home', df_['homeTeamAbbr'], df_['visitorTeamAbbr']))
        .drop(columns=['teamReturner', 'team', 'homeTeamAbbr', 'visitorTeamAbbr'])
        # calculate distance to returner
        .assign(dist=lambda df_: np.sqrt((df_['x']-df_['x_returner'])**2 + (df_['y']-df_['y_returner'])**2))
        # attach distance order within given play
        .sort_values(['gameId','playId','frameId','dist'])
        # create features
        .assign(distOrder=lambda df_: df_.groupby(['gameId','playId','frameId']).cumcount()+1,
                timeToClose=lambda df_: df_.dist / np.maximum(df_.s, 0.01),
                upGutLeverage=lambda df_: np.abs(df_.y - df_.y_returner)
                )
        # filter to closest n defenders
        .query('distOrder <= ' + str(n_defenders))
        # pivot so that one line is for a given gameId/playId/frameId
        .pivot(index=['gameId','playId','frameId'],columns=['distOrder'], values=['dist','timeToClose','upGutLeverage'])
        )
    # rename the pivoted columns dynamically
    return_df.columns = return_df.columns.to_flat_index()
    # take gameId, playId, and frameId out of index
    return_df.reset_index(inplace=True)

    return return_df

def feat_returnerSpeedAtCatch(track_df, play_df):
    return (
        track_df
        # get location data of returner for each frame for modeling
        .merge(play_df.loc[~play_df.returnerId.astype(str).str.contains(';'), ## only consider plays with 1 returner
                            ['gameId','playId','returnerId']].astype({"returnerId": float}),
                how='inner',
                left_on=['gameId','playId','nflId'],
                right_on=['gameId','playId','returnerId'])
        .assign(timeToCatch = lambda df_: feat_timeToCatch(df_)['timeToCatch'])
        # filter to time of catch
        .query('timeToCatch == 0.0')
        .assign(s_dwnfld = lambda df_: df_.s * -np.sin(np.deg2rad(df_.dir)))  # 0-180 degrees are backwards
        # filter to applicable columns
        .filter(['gameId','playId', 's_dwnfld'])
    )