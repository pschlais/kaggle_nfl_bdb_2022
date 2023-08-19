import pandas as pd
import numpy as np
import nflutil

# ===== PREP FUNCTIONS ========================= #
def prep_get_modeling_frames(track_df: pd.DataFrame, play_df: pd.DataFrame, pff_df: pd.DataFrame, play_end_event_name: str='punt_received') -> pd.DataFrame:
    
    # get the applicable play filter depending on the desired play type
    if play_end_event_name == 'punt_received':
        play_filter = (play_df.specialTeamsPlayType=='Punt') & (play_df.specialTeamsResult=='Return') & (~play_df.kickReturnYardage.isna())
    elif play_end_event_name == 'fair_catch':
        play_filter = (play_df.specialTeamsPlayType=='Punt') & (play_df.specialTeamsResult=='Fair Catch')
    else:
        raise ValueError(f'Invalid input for play_end_event_name in nfl_bdb22.prep_get_modeling_frames(): expected "punt_received" or "fair_catch". Input was {play_end_event_name}')

    return (
        track_df
        # (1) filter down to punt plays with a return and recordable yardage
        .merge(play_df.loc[play_filter, 
                           ['gameId','playId']],
               how='inner',
               on=['gameId','playId']
              )
        # (2) only consider plays with a single returner (simplifies analysis - 1 target for defenders to converge on)
        .merge(play_df.loc[~play_df.returnerId.astype(str).str.contains(';'), ## only consider plays with 1 returner
                            ['gameId','playId']],
                how='inner',
                left_on=['gameId','playId'],
                right_on=['gameId','playId']
              )
        # (3) filter down to clean catches
        .merge(pff_df.loc[pff_df.kickContactType=='CC', ['gameId','playId']],
               how='inner',
               on=['gameId','playId']
              )
        # (4) filter to all frames between the punt and the catch
        .merge(
            # Dataframe of: gameId, playId, punt (frameId), punt_received (frameId)
            (nflutil.get_frame_of_event(track_df, 
                                   ['punt', play_end_event_name])
             .pivot(index=['gameId','playId'], columns='event', values='frameId')
             .dropna()  # remove any rows where punt OR punt_received events are not present (both means a valid play)
             .rename(columns={'punt': 'puntFrameId', play_end_event_name: 'puntReceivedFrameId'})
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

def prep_remove_low_hangtime_punts(track_df: pd.DataFrame, hangtime_thresh: float) -> pd.DataFrame:
    """This function assumes that the track_df has already been filtered to punt plays."""
    plays_to_remove = (
        # get hangtime for each play
        feat_timeToCatch(track_df)
        .groupby(['gameId','playId'])['timeToCatch'].max()
        .reset_index()
        # filter hangtime
        .query(f'timeToCatch <= {hangtime_thresh}')
        # create new column for (gameId, playId)
        .assign(gamePlayKey = lambda df_: list(zip(df_.gameId, df_.playId)))
        ['gamePlayKey']
        .to_list()
    )
    if len(plays_to_remove) > 0:
        print(f'INFO: The following {len(plays_to_remove)} plays were removed for hangtime <= {hangtime_thresh}: {plays_to_remove}')
        return nflutil.remove_abnormal_plays(track_df, plays_to_remove)
    else: # no plays to remove
        return track_df

# ==== FEATURE GENERATION FUNCTIONS =========================================
def feat_timeToCatch(track_df: pd.DataFrame, catch_type: str='punt_received') -> pd.DataFrame:
    # adds timeToCatch feature to dataframe
    
    # get the applicable play filter depending on the desired play type
    if catch_type not in ['punt_received', 'fair_catch']:
        raise ValueError(f'Invalid input for catch_type in nfl_bdb22.feat_timeToCatch(): expected "punt_received" or "fair_catch". Input was {catch_type}')
    
    return (
        track_df
        .merge(
            # Dataframe of: gameId, playId, punt_received (frameId)
            (nflutil.get_frame_of_event(track_df, catch_type)
             .pivot(index=['gameId','playId'], columns='event', values='frameId')
             .rename(columns={catch_type: 'puntReceivedFrameId'})
             .astype('int64')
             .reset_index()
                ),
            how='left',
            on=['gameId','playId']
        )
        .assign(timeToCatch=lambda df_: (df_['puntReceivedFrameId'] - df_['frameId'])/10)
        .drop(columns=['puntReceivedFrameId'])
    )


def feat_byDefender(track_df: pd.DataFrame, play_df: pd.DataFrame, game_df: pd.DataFrame, n_defenders: int = 4, catch_type: str='punt_received') -> pd.DataFrame:
    
    return_df =  (
        track_df
        # add time to catch as a baseline value for feature generation
        # -- feat_timeToCatch modifies the indexes because of the merge(), so assign() doesn't work properly if in Series format
        .assign(timeToCatch = lambda df_: feat_timeToCatch(track_df=df_, catch_type=catch_type)['timeToCatch'].to_numpy()) 
        # get location data of returner for each frame for modeling
        .merge(play_df.loc[~play_df.returnerId.astype(str).str.contains(';'), ## only consider plays with 1 returner
                            ['gameId','playId','returnerId']].astype({"returnerId": float}),
                how='inner',
                left_on=['gameId','playId','nflId'],
                right_on=['gameId','playId','returnerId'])
        .filter(['gameId','playId','frameId','timeToCatch','team','x','y'])
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
                upGutLeverage=lambda df_: np.abs(df_.y - df_.y_returner),
                willReachFactor=lambda df_: (df_.timeToClose - df_.timeToCatch) / np.maximum(0.1, df_.timeToCatch),
                willReach=lambda df_: np.where(df_.timeToClose <= df_.timeToCatch, 1, 0),
                reachWithin5=lambda df_: np.where((df_.dist - df_.s * df_.timeToCatch) <= 5, 1, 0),
                reachWithin10=lambda df_: np.where((df_.dist - df_.s * df_.timeToCatch) <= 10, 1, 0),
                reachWithin20=lambda df_: np.where((df_.dist - df_.s * df_.timeToCatch) <= 20, 1, 0),
                reachWithin30=lambda df_: np.where((df_.dist - df_.s * df_.timeToCatch) <= 30, 1, 0)
                )
        # attach counts to each record for pivot
        .pipe(lambda df_: df_.drop(columns=['willReach','reachWithin5','reachWithin10','reachWithin20','reachWithin30']).merge(
                # get counts for each "within" count
                df_.groupby(['gameId','playId','frameId'])[['willReach','reachWithin5','reachWithin10','reachWithin20','reachWithin30']].sum(),
                how='inner',
                on=['gameId','playId','frameId']
                )
            )
        # filter to closest n defenders
        .query('distOrder <= ' + str(n_defenders))
        # pivot so that one line is for a given gameId/playId/frameId
        .pivot(index=['gameId','playId','frameId','timeToCatch','willReach','reachWithin5','reachWithin10','reachWithin20','reachWithin30'], 
               columns=['distOrder'], 
               values=['dist','timeToClose','upGutLeverage','willReachFactor']
               )
        )
    # rename the pivoted columns dynamically
    return_df.columns = return_df.columns.to_flat_index()
    # take gameId, playId, and frameId out of index
    return_df.reset_index(inplace=True)

    return return_df

def feat_returnerLateralSpeed(track_df: pd.DataFrame, play_df: pd.DataFrame) -> pd.DataFrame:
    return (
        track_df
        # get location data of returner for each frame for modeling
        .merge(play_df.loc[~play_df.returnerId.astype(str).str.contains(';'), ## only consider plays with 1 returner
                            ['gameId','playId','returnerId']].astype({"returnerId": float}),
                how='inner',
                left_on=['gameId','playId','nflId'],
                right_on=['gameId','playId','returnerId'])
        .assign(s_lateral = lambda df_: df_.s * np.abs(np.cos(np.deg2rad(df_.dir))))  # speed only, don't care about which sideline direction
        # filter to applicable columns
        .filter(['gameId','playId', 'frameId', 's_lateral'])
    )

def feat_returnerDownfieldSpeed(track_df: pd.DataFrame, play_df: pd.DataFrame) -> pd.DataFrame:
    return (
        track_df
        # get location data of returner for each frame for modeling
        .merge(play_df.loc[~play_df.returnerId.astype(str).str.contains(';'), ## only consider plays with 1 returner
                            ['gameId','playId','returnerId']].astype({"returnerId": float}),
                how='inner',
                left_on=['gameId','playId','nflId'],
                right_on=['gameId','playId','returnerId'])
        .assign(s_dwnfld = lambda df_: df_.s * -np.sin(np.deg2rad(df_.dir)))  # 0-180 degrees are backwards/towards own endzone
        # filter to applicable columns
        .filter(['gameId','playId', 'frameId', 's_dwnfld'])
    )

def feat_returnerSpeed(track_df: pd.DataFrame, play_df: pd.DataFrame) -> pd.DataFrame:
    return (
        track_df
        # get location data of returner for each frame for modeling
        .merge(play_df.loc[~play_df.returnerId.astype(str).str.contains(';'), ## only consider plays with 1 returner
                            ['gameId','playId','returnerId']].astype({"returnerId": float}),
                how='inner',
                left_on=['gameId','playId','nflId'],
                right_on=['gameId','playId','returnerId'])
        .assign(s_abs=lambda df_: df_.s)
        # filter to applicable columns
        .filter(['gameId','playId', 'frameId', 's_abs'])
    )

def feat_returnerDistFromSideline(track_df: pd.DataFrame, play_df: pd.DataFrame) -> pd.DataFrame:
    return (
        track_df
        # get location data of returner for each frame for modeling
        .merge(play_df.loc[~play_df.returnerId.astype(str).str.contains(';'), ## only consider plays with 1 returner
                            ['gameId','playId','returnerId']].astype({"returnerId": float}),
                how='inner',
                left_on=['gameId','playId','nflId'],
                right_on=['gameId','playId','returnerId'])
        # distance to closest sideline
        .assign(distFromSideline=lambda df_: np.where(df_['y'] < nflutil.FIELD_SIZE_Y / 2, df_['y'], nflutil.FIELD_SIZE_Y - df_['y']))
        # filter to applicable columns
        .filter(['gameId','playId', 'frameId', 'distFromSideline'])
    )

### This may not be a relevant function - for fair catch analysis this is an unknown future value
# def feat_returnerSpeedAtCatch(track_df: pd.DataFrame, play_df: pd.DataFrame) -> pd.DataFrame:
#     return (
#         track_df
#         # get location data of returner for each frame for modeling
#         .merge(play_df.loc[~play_df.returnerId.astype(str).str.contains(';'), ## only consider plays with 1 returner
#                             ['gameId','playId','returnerId']].astype({"returnerId": float}),
#                 how='inner',
#                 left_on=['gameId','playId','nflId'],
#                 right_on=['gameId','playId','returnerId'])
#         .assign(timeToCatch = lambda df_: feat_timeToCatch(df_)['timeToCatch'])
#         # filter to time of catch
#         .query('timeToCatch == 0.0')
#         .assign(s_dwnfld = lambda df_: df_.s * -np.sin(np.deg2rad(df_.dir)))  # 0-180 degrees are backwards
#         # filter to applicable columns
#         .filter(['gameId','playId', 's_dwnfld_at_catch'])
#     )

def model_create_features(clean_track_df: pd.DataFrame, play_df: pd.DataFrame, game_df: pd.DataFrame, n_defenders: int, catch_type: str='punt_received') -> pd.DataFrame:
    modeling_feature_df = (
    clean_track_df
    # compress to gameId, playId, frameId to build features onto
    [['gameId','playId','frameId']]
    .drop_duplicates()
    ## === ADD FEATURES =====================
    # defender stats for n-closest defenders at given frame - also compress output to each record = game-play-frame
    .merge(feat_byDefender(track_df=clean_track_df, play_df=play_df, game_df=game_df, n_defenders=n_defenders, catch_type=catch_type),
           how='inner',
           on=['gameId','playId','frameId']
          )
    # returner speed
    .merge(feat_returnerSpeed(track_df=clean_track_df, play_df=play_df),
           how='inner',
           on=['gameId','playId','frameId'])
    # returner downfield speed
    .merge(feat_returnerDownfieldSpeed(track_df=clean_track_df, play_df=play_df),
           how='inner',
           on=['gameId','playId','frameId'])
    # returner lateral speed
    .merge(feat_returnerLateralSpeed(track_df=clean_track_df, play_df=play_df),
           how='inner',
           on=['gameId','playId','frameId'])
    # returner distance to closest sideline
    .merge(feat_returnerDistFromSideline(track_df=clean_track_df, play_df=play_df),
           how='inner',
           on=['gameId','playId','frameId'])
    )
    return modeling_feature_df