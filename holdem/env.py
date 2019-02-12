# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 Aleksander Beloi (beloi.alex@gmail.com)
# Copyright (c) 2018 Sam Wenke (samwenke@gmail.com)
# Copyright (c) 2019 Ingvar Lond (ingvar.lond@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from enum import IntEnum

from gym import Env, error, spaces, utils
from gym.utils import seeding

from treys import Card, Deck, Evaluator

from .player import Player
from .utils import hand_to_str, format_action, action_table, community_table, player_table

class Street(IntEnum):
    NOT_STARTED = 0
    PREFLOP = 1
    FLOP = 2
    TURN = 3
    RIVER = 4
    SHOWDOWN = 5

class TexasHoldemEnv(Env, utils.EzPickle):
  BLIND_INCREMENTS = [[10,25], [25,50], [50,100], [75,150], [100,200],
                      [150,300], [200,400], [300,600], [400,800], [500,10000],
                      [600,1200], [800,1600], [1000,2000]]

  def __init__(self, n_seats, max_limit=100000, debug=False):
    n_suits = 4                     # s,h,d,c
    n_ranks = 13                    # 2,3,4,5,6,7,8,9,T,J,Q,K,A
    n_community_cards = 5           # flop, turn, river
    n_pocket_cards = 2
    n_stud = 5

    self.n_seats = n_seats

    self._blind_index = 0
    [self._smallblind, self._bigblind] = TexasHoldemEnv.BLIND_INCREMENTS[0]
    self._deck = Deck()
    self._evaluator = Evaluator()

    self.community = []
    self._round = Street.NOT_STARTED
    self._button = 0

    self._side_pots = [0] * n_seats
    self._current_sidepot = 0 # index of _side_pots
    self._totalpot = 0
    self._tocall = 0
    self._lastraise = 0
    self._number_of_hands = 0

    # fill seats with dummy players
    self._seats = [Player(i, stack=0, emptyplayer=True) for i in range(n_seats)]
    self.emptyseats = n_seats
    self._player_dict = {}
    self._current_player = None
    self._debug = debug
    self._last_player = None
    self._last_actions = None

    self.observation_space = spaces.Tuple([
      spaces.Tuple([                # players
        spaces.MultiDiscrete([
          1,                   # emptyplayer
          n_seats - 1,         # seat
          max_limit,           # stack
          1,                   # is_playing_hand
          max_limit,           # handrank
          1,                   # playedthisround
          1,                   # is_betting
          1,                   # isallin
          max_limit,           # last side pot
        ]),
        spaces.Tuple([
          spaces.MultiDiscrete([    # hand
            n_suits,          # suit, can be negative one if it's not avaiable.
            n_ranks,          # rank, can be negative one if it's not avaiable.
          ])
        ] * n_pocket_cards)
      ] * n_seats),
      spaces.Tuple([
        spaces.Discrete(n_seats - 1), # big blind location
        spaces.Discrete(max_limit),   # small blind
        spaces.Discrete(max_limit),   # big blind
        spaces.Discrete(max_limit),   # pot amount
        spaces.Discrete(max_limit),   # last raise
        spaces.Discrete(max_limit),   # minimum amount to raise
        spaces.Discrete(max_limit),   # how much needed to call by current player.
        spaces.Discrete(n_seats - 1), # current player seat location.
        spaces.MultiDiscrete([        # community cards
          n_suits - 1,          # suit
          n_ranks - 1,          # rank
          1,                     # is_flopped
        ]),
      ] * n_stud),
    ])

    self.action_space = spaces.Tuple([
      spaces.MultiDiscrete([
        3,                     # action_id
        max_limit,             # raise_amount
      ]),
    ] * n_seats)

  def seed(self, seed=None):
    _, seed = seeding.np_random(seed)
    return [seed]

  def add_player(self, seat_id, stack=2000):
    """Add a player to the environment seat with the given stack (chipcount)"""
    player_id = seat_id
    if player_id not in self._player_dict:
      new_player = Player(player_id, stack=stack, emptyplayer=False)
      if self._seats[player_id].emptyplayer:
        self._seats[player_id] = new_player
        new_player.set_seat(player_id)
      else:
        raise error.Error('Seat already taken.')
      self._player_dict[player_id] = new_player
      self.emptyseats -= 1

  def remove_player(self, seat_id):
    """Remove a player from the environment seat."""
    player_id = seat_id
    try:
      idx = self._seats.index(self._player_dict[player_id])
      self._seats[idx] = Player(0, stack=0, emptyplayer=True)
      del self._player_dict[player_id]
      self.emptyseats += 1
    except ValueError:
      pass

  def reset(self):
    self._reset_game()
    self._number_of_hands = 1
    [self._smallblind, self._bigblind] = TexasHoldemEnv.BLIND_INCREMENTS[0]
    if len(self._player_dict) >= 2:
      players = [p for p in self._seats if p.playing_hand]
      self._new_round()
      self._current_player = self._first_to_act(players)
      self._post_smallblind(self._current_player)
      self._current_player = self._next(players, self._current_player)
      self._post_bigblind(self._current_player)
      self._current_player = self._next(players, self._current_player)
      self._tocall = self._bigblind
      self._deal_next_round()
      self._folded_players = []
    return self._get_current_reset_returns()

  def step(self, actions):
    """
    CHECK = 0
    CALL = 1
    RAISE = 2
    FOLD = 3

    RAISE_AMT = [0, minraise]
    """
    if len(actions) != len(self._seats):
      raise error.Error('actions must be same shape as number of seats.')

    if self._current_player is None:
      raise error.Error('Round cannot be played without 2 or more players.')

    if self._round == Street.SHOWDOWN:
      raise error.Error('Rounds already finished, needs to be reset.')

    players = [p for p in self._seats if p.playing_hand]
    if len(players) == 1:
      raise error.Error('Round cannot be played with one player.')

    self._last_player = self._current_player
    self._last_actions = actions

    if any([p.isallin == False for p in players]):
      if self._current_player.isallin:
        self._current_player = self._next(players, self._current_player)
        return self._get_current_step_returns(False)
      move = self._current_player.player_move(
              self._output_state(self._current_player), actions[self._current_player.player_id])
      if self._debug:
        print('Player', self._current_player.player_id, move)
      if move[0] != 'fold':
        self._player_bet(self._current_player, move[1])
      if move[0] == 'raise':
        for p in players:
          if p != self._current_player:
            p.playedthisround = False

      if move[0] == 'fold':
        self._current_player.playing_hand = False
        folded_player = self._current_player
        players.remove(folded_player)
        self._folded_players.append(folded_player)
      else:
        self._current_player = self._next(players, self._current_player)
      self._current_player.playedthisround
        # break if a single player left <<-- already will be resolved in next check, if we folded then there must be a bet and the other player has acted
        #if len(players) == 1:
          #self._resolve(players)
    if all([player.playedthisround for player in players]):
      self._resolve(players)

    terminal = False
    if all([player.isallin for player in players]):
      while self._round < Street.SHOWDOWN:
        self._deal_next_round()
    if self._round == Street.SHOWDOWN or len(players) == 1:
      terminal = True
      self._resolve_round(players)
    return self._get_current_step_returns(terminal)

  def render(self, mode='human', close=False):
    print('total pot: {}'.format(self._totalpot))
    if self._last_actions is not None:
      pid = self._last_player.player_id
      print('last action by player {}:'.format(pid))
      print(format_action(self._last_player, self._last_actions[pid]))

    (player_states, community_states) = self._get_current_state()
    (player_infos, player_hands) = zip(*player_states)
    (community_infos, community_cards) = community_states
    blinds_idxs = self._get_blind_indexes(community_infos)

    print('community:')
    print('-' + hand_to_str(community_cards))
    print('players:')
    for idx, hand in enumerate(player_hands):
      position_string = self._get_blind_str(blinds_idxs, idx)
      print('{} {}{}stack: {}'.format(idx, position_string, hand_to_str(hand), self._seats[idx].stack))

  def _get_blind_str(self, blinds_idxs, idx):
    if idx == blinds_idxs[0]:
      return "SB"
    elif idx == blinds_idxs[1]:
      return "BB"
    else:
      return "  "

  def _get_blind_indexes(self, community_infos):
    idx = community_infos[community_table.BUTTON_POS]
    # If more than 2 players playing, SB is next from BTN, else BTN is SB
    if len([s for s in self._seats if not s.sitting_out and not s.emptyplayer]) > 2:
      idx = (idx + 1) % len(self._seats)
    sb_idx = -1
    while True:
      while self._seats[idx].sitting_out or self._seats[idx].emptyplayer:
        idx = (idx + 1) % len(self._seats)
      if sb_idx == -1:
        sb_idx = idx
      else:
        return (sb_idx, idx)
      idx = (idx + 1) % len(self._seats)

  def _resolve(self, players):
    self._current_player = self._first_to_act(players)
    self._resolve_sidepots(players + self._folded_players)
    self._new_round()
    self._deal_next_round()
    if self._debug:
      print('totalpot', self._totalpot)

  def _deal_next_round(self):
    if self._round == Street.NOT_STARTED:
      self._deal()
    elif self._round == Street.PREFLOP:
      self._flop()
    elif self._round == Street.FLOP:
      self._turn()
    elif self._round == Street.TURN:
      self._river()
    self._round += 1

  def _increment_blinds(self):
    self._blind_index = min(self._blind_index + 1, len(TexasHoldemEnv.BLIND_INCREMENTS) - 1)
    [self._smallblind, self._bigblind] = TexasHoldemEnv.BLIND_INCREMENTS[self._blind_index]

  def _post_smallblind(self, player):
    if self._debug:
      print('player ', player.player_id, 'small blind', self._smallblind)
    self._player_bet(player, self._smallblind)
    player.blind = self._smallblind
    player.playedthisround = False

  def _post_bigblind(self, player):
    if self._debug:
      print('player ', player.player_id, 'big blind', self._bigblind)
    self._player_bet(player, self._bigblind)
    player.playedthisround = False
    player.blind = self._bigblind
    self._lastraise = self._bigblind

  def _player_bet(self, player, total_bet):
    # relative_bet is how much _additional_ money is the player betting this turn,
    # on top of what they have already contributed
    # total_bet is the total contribution by player to pot in this round
    relative_bet = min(player.stack, total_bet - player.currentbet)
    player.bet(total_bet)

    self._totalpot += relative_bet
    self._tocall = max(self._tocall, total_bet)
    if self._tocall > 0:
      self._tocall = max(self._tocall, self._bigblind)
    self._lastraise = max(self._lastraise, relative_bet  - self._lastraise)

  def _first_to_act(self, players):
    if self._round == Street.NOT_STARTED and len(players) == 2:
      return self._next(sorted(
          players + [self._seats[self._button]], key=lambda x:x.get_seat()),
          self._seats[self._button])
    try:
      first = [player for player in players if player.get_seat() > self._button][0]
    except IndexError:
      first = players[0]
    return first

  def _next(self, players, current_player):
    idx = players.index(current_player)
    return players[(idx+1) % len(players)]

  def _deal(self):
    for player in self._seats:
      if player.playing_hand:
        player.hand = self._deck.draw(2)

  def _flop(self):
    self.community = self._deck.draw(3)

  def _turn(self):
    self.community.append(self._deck.draw(1))

  def _river(self):
    self.community.append(self._deck.draw(1))

  def _resolve_sidepots(self, players_playing):
    players = [p for p in players_playing if p.currentbet]
    if self._debug:
      print('current bets: ', [p.currentbet for p in players])
      print('playing hand: ', [p.playing_hand for p in players])
    if not players:
      return
    try:
      smallest_bet = min([p.currentbet for p in players if p.playing_hand])
    except ValueError:
      for p in players:
        self._side_pots[self._current_sidepot] += p.currentbet
        p.currentbet = 0
      return

    smallest_players_allin = [p for p, bet in zip(players, [p.currentbet for p in players]) if bet == smallest_bet and p.isallin]

    for p in players:
      self._side_pots[self._current_sidepot] += min(smallest_bet, p.currentbet)
      p.currentbet -= min(smallest_bet, p.currentbet)
      p.lastsidepot = self._current_sidepot

    if smallest_players_allin:
      self._current_sidepot += 1
      self._resolve_sidepots(players)
    if self._debug:
      print('sidepots: ', self._side_pots)

  def _new_round(self):
    for player in self._player_dict.values():
      player.currentbet = 0
      player.playedthisround = False
    self._tocall = 0
    self._lastraise = 0

  def _resolve_round(self, players):
    if len(players) == 1:
      players[0].refund(sum(self._side_pots))
      self._totalpot = 0
    else:
      # compute hand ranks
      for player in players:
        player.handrank = self._evaluator.evaluate(player.hand, self.community)

      # trim side_pots to only include the non-empty side pots
      temp_pots = [pot for pot in self._side_pots if pot > 0]

      # compute who wins each side pot and pay winners
      for pot_idx,_ in enumerate(temp_pots):
        # find players involved in given side_pot, compute the winner(s)
        pot_contributors = [p for p in players if p.lastsidepot >= pot_idx]
        winning_rank = min([p.handrank for p in pot_contributors])
        winning_players = [p for p in pot_contributors if p.handrank == winning_rank]

        for player in winning_players:
          split_amount = int(self._side_pots[pot_idx]/len(winning_players))
          if self._debug:
            print('Player', player.player_id, 'wins side pot (', int(self._side_pots[pot_idx]/len(winning_players)), ')')
          player.refund(split_amount)
          self._side_pots[pot_idx] -= split_amount

        # any remaining chips after splitting go to the winner in the earliest position
        if self._side_pots[pot_idx]:
          earliest = self._first_to_act([player for player in winning_players])
          earliest.refund(self._side_pots[pot_idx])

  def _reset_game(self):
    self._round = Street.NOT_STARTED
    playing = 0
    for player in self._seats:
      if not player.emptyplayer and not player.sitting_out:
        player.reset_hand()
        playing += 1
    self.community = []
    self._current_sidepot = 0
    self._totalpot = 0
    self._side_pots = [0] * len(self._seats)
    self._deck.shuffle()

    if playing:
      self._button = (self._button + 1) % len(self._seats)
      while not self._seats[self._button].playing_hand:
        self._button = (self._button + 1) % len(self._seats)

  def _output_state(self, current_player):
    return {
      'players': [player.player_state() for player in self._seats],
      'community': self.community,
      'my_seat': current_player.get_seat(),
      'pocket_cards': current_player.hand,
      'pot': self._totalpot,
      'button': self._button,
      'tocall': (self._tocall),
      'stack': current_player.stack,
      'bigblind': self._bigblind,
      'player_id': current_player.player_id,
      'lastraise': self._lastraise,
      'minraise': max(self._bigblind, self._lastraise + self._tocall),
    }

  def _pad(self, l, n, v):
    if (not l) or (l is None):
      l = []
    return l + [v] * (n - len(l))

  def _get_current_state(self):
    player_states = []
    for player in self._seats:
      player_features = [
        int(player.emptyplayer),
        int(player.get_seat()),
        int(player.stack),
        int(player.playing_hand),
        int(player.handrank),
        int(player.playedthisround),
        int(player.betting),
        int(player.isallin),
        int(player.lastsidepot),
      ]
      player_states.append((player_features, self._pad(player.hand, 2, -1)))
    community_states = ([
      int(self._button),
      int(self._smallblind),
      int(self._bigblind),
      int(self._totalpot),
      int(self._lastraise),
      int(max(self._bigblind, self._lastraise + self._tocall)),
      int(self._tocall - self._current_player.currentbet),
      int(self._current_player.player_id),
    ], self._pad(self.community, 5, -1))
    return (tuple(player_states), community_states)

  def _get_current_reset_returns(self):
    return self._get_current_state()

  def _get_current_step_returns(self, terminal):
    obs = self._get_current_state()
    rew = [player.stack - player.hand_starting_stack + player.blind if terminal else 0 for player in self._seats]
    info = {}
    info['money_won'] = self._seats[0].stack - self._seats[0].hand_starting_stack if terminal else 0
    return obs, rew, terminal, info
