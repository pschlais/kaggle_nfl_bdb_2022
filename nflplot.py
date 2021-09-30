import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import nflutil


# --------- CONSTANTS --------------------------- ####
FIELD_SIZE_X = nflutil.FIELD_SIZE_X  # yards, field goal to field goal (back of endzones)
FIELD_SIZE_Y = nflutil.FIELD_SIZE_Y  # yards, sideline to sideline
TEAM_COLORS = nflutil.TEAM_COLORS  # dict of {'Team': (R,G,B)} where RGB values are in range 0-1


# -------- CLASSES ----------------------------------- ###
class PlayAnimation:
    """
    Creates an animation for a given play. Stored in "animation" property of object.
    """

    def __init__(self, track_df, play_df, game_df, game_id, play_id, fig_x_dim=16):
        """
        :param track_df: DataFrame from 'weekX.csv' files
        :param play_df: DataFrame from 'plays.csv' file
        :param game_df: DataFrame from 'games.csv' file
        :param game_id: gameId of play to plot
        :param play_id: playId of play to plot
        :param fig_x_dim: x-dimension size of output figure
        """
        # Filter down DataFrames to the specific play (tracks, game, and play info)
        track_df = track_df[(track_df.gameId == game_id) & (track_df.playId == play_id)].sort_values('frameId')
        track_df = track_df.drop_duplicates()
        game_df = game_df[game_df.gameId == game_id].iloc[0]
        play_df = play_df[(play_df.gameId == game_id) & (play_df.playId == play_id)].iloc[0]

        # Generate the plot title
        # Example: '(2Q 10:33) ATL Possession, 3rd & 5 at ATL 32 [09-08-2018, ATL @ KC] [gameId=1234, playId=123]'
        down_dict = {1: '1st', 2: '2nd', 3: '3rd', 4: '4th'}
        togo = str(play_df.yardsToGo) if (play_df.yardsToGo != play_df.yardlineNumber) else 'Goal'
        self._title = (f'({play_df.quarter}Q {play_df.gameClock[:-3]}) {play_df.possessionTeam} Possession, '
                       + f'{down_dict[int(play_df.down)]} & {togo} at {play_df.yardlineSide} {play_df.yardlineNumber} '
                       + f'[{game_df.gameDate}, {game_df.visitorTeamAbbr} @ {game_df.homeTeamAbbr}] '
                       + f'[gameId={game_id}, playId={play_id}]')

        # first down marker
        self._first_down_distance = play_df.yardsToGo

        # set properties
        self._num_players = len(track_df.nflId.unique())
        self._team_colors = {'home': TEAM_COLORS[game_df.homeTeamAbbr],
                             'away': TEAM_COLORS[game_df.visitorTeamAbbr]}
        self._frame_data = track_df
        self._frame_ids = track_df['frameId'].copy().sort_values().unique()

        # plotting handles - scale figure to be equal aspect ratio of field
        self._fig = plt.figure(figsize=(fig_x_dim, fig_x_dim * (FIELD_SIZE_Y / FIELD_SIZE_X)))
        self._ax_base = self._fig.gca()  # base Axes (field)
        self._ax_play = self._ax_base.twinx()  # play Axes (players, football) on top of the base field markers

        # scatter plot entities - placeholders
        self._scat_fb = self._ax_play.scatter([], [], s=100, color='brown')
        self._scat_home = self._ax_play.scatter([], [], s=400, color=self._team_colors['home']['main'],
                                                edgecolors=self._team_colors['home']['secondary'])
        self._scat_away = self._ax_play.scatter([], [], s=400, color=self._team_colors['away']['main'],
                                                edgecolors=self._team_colors['away']['secondary'])

        # containers for player plot entities
        self._scat_position_list = []
        self._scat_number_list = []
        self._scat_name_list = []
        self._plot_track_list = []

        # output animation
        self.animation = animation.FuncAnimation(self._fig, self.update, frames=self._frame_ids, interval=100,
                                                 init_func=self.base_plot)
        # plt.close()

    @staticmethod
    def set_axis_plots(ax, max_x, max_y):
        # don't show axis labels (football field is shown)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # set limits to football field edges
        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_y])
        # ensure the X-Y plot values are 1:1 aspect ratio
        ax.axes.set_aspect('equal')

    def base_plot(self):
        # set up the Axes objects
        self.set_axis_plots(self._ax_base, FIELD_SIZE_X, FIELD_SIZE_Y)
        self.set_axis_plots(self._ax_play, FIELD_SIZE_X, FIELD_SIZE_Y)

        # set the title
        self._ax_base.set_title(self._title)

        # plot the line of scrimmage
        los = self._frame_data.loc[(self._frame_data.team == 'football'), 'x'].iloc[0]
        self._ax_base.axvline(los, color='k', linestyle='--')

        # plot the first down marker
        dir_factor = 1 if self._frame_data.playDirection.iloc[0] == 'right' else -1
        self._ax_base.axvline(los + dir_factor * self._first_down_distance, color='darkorange', linestyle='-')

        # plot the sidelines
        for side_line in [0, FIELD_SIZE_Y]:
            self._ax_base.plot([0, FIELD_SIZE_X], [side_line, side_line], color='k', linestyle='-', alpha=0.8)

        # plot the line markers across the field in 10-yard increments
        for yd_line in range(0, int(FIELD_SIZE_X) + 1, 10):
            self._ax_base.plot([yd_line, yd_line], [0, FIELD_SIZE_Y], color='k', linestyle='-', alpha=0.2)

        # plot all hash marks not on the lines across the field (converted yard lines to x-coordinates)
        hash_x = np.array([i for i in range(1, 100) if i % 10 != 0]) + 10
        for yd_line in hash_x:
            # home side inner hashes
            self._ax_base.plot([yd_line, yd_line], [22.916, 23.583], color='k', alpha=0.2)
            # away side inner hashes
            self._ax_base.plot([yd_line, yd_line], [29.75, 30.416], color='k', alpha=0.2)

        # endzone markers and 50 yard line with darker lines
        for ez_line in [0, 10, 60, 110, 120]:
            self._ax_base.plot([ez_line, ez_line], [0, FIELD_SIZE_Y], color='k', linestyle='-', alpha=0.8)

        # add placeholders for the player-specific plot items
        for _ in range(self._num_players):
            # position text (QB, RB, etc.)
            self._scat_position_list.append(
                self._ax_play.text(0, 0, '', horizontalalignment='center', verticalalignment='center'))
            # player numbers (12, 88, etc.)
            self._scat_number_list.append(
                self._ax_play.text(0, 0, '', horizontalalignment='center', verticalalignment='center'))
            # player names
            self._scat_name_list.append(
                self._ax_play.text(0, 0, '', horizontalalignment='center', verticalalignment='center'))
            # player track histories from the start of the play
            self._plot_track_list.append(self._ax_play.plot([], [], alpha=0.5))

        # for animation construction, return all of the plot objects that can change between frames
        return (self._scat_fb, self._scat_home, self._scat_away, *self._scat_position_list, *self._scat_number_list,
                *self._scat_name_list, *self._plot_track_list)

    def update(self, anim_frame):
        # cumulative tracking data for track history
        hist_df = self._frame_data[self._frame_data.frameId <= anim_frame]
        # current frame tracking data
        pos_df = self._frame_data[self._frame_data.frameId == anim_frame]

        # plot the dots for home team, away team, and football
        for label in pos_df.team.unique():
            label_data = pos_df[pos_df.team == label]

            if label == 'football':
                self._scat_fb.set_offsets(np.hstack([label_data.x, label_data.y]))
            elif label == 'home':
                self._scat_home.set_offsets(np.vstack([label_data.x, label_data.y]).T)
            elif label == 'away':
                self._scat_away.set_offsets(np.vstack([label_data.x, label_data.y]).T)

        # get tracking data for current play for all players (i.e. not the football)
        player_df = pos_df[pos_df.jerseyNumber.notnull()]

        # loop over players in current frame tracking data to plot
        for (index, player) in player_df.reset_index().iterrows():
            # position of each player
            self._scat_position_list[index].set_position((player.x, player.y))  # inside dot
            self._scat_position_list[index].set_text(player.position)  # position (QB, RB, etc.)
            self._scat_position_list[index].set_color(self._team_colors[player.team]['secondary'])  # text color
            # number of each player
            self._scat_number_list[index].set_position((player.x, player.y + 1.9))  # above dot
            self._scat_number_list[index].set_text(int(player.jerseyNumber))  # number
            self._scat_number_list[index].set_color(self._team_colors[player.team]['main'])  # text color
            # name of each player
            self._scat_name_list[index].set_position((player.x, player.y - 1.9))  # below dot
            self._scat_name_list[index].set_text(player.displayName.split()[-1])  # last name only
            self._scat_name_list[index].set_color(self._team_colors[player.team]['main'])  # text color
            # track of each player from the previous frames
            self._plot_track_list[index][0].set_data(hist_df[hist_df.nflId == player.nflId]['x'],
                                                     hist_df[hist_df.nflId == player.nflId]['y'])
            self._plot_track_list[index][0].set_color(self._team_colors[player.team]['main'])

            # player_speed = player.s

        # return the updated plot objects
        return (self._scat_fb, self._scat_home, self._scat_away, *self._scat_position_list, *self._scat_number_list,
                *self._scat_name_list, *self._plot_track_list)

# -------- FUNCTIONS --------------------------------- ###
