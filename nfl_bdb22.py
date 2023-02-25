import pandas as pd
import numpy as np
import nflutil

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
