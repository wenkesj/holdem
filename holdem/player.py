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
from random import randint

from gym import error

from treys import Card


class Player(object):

  CHECK = 0
  CALL = 1
  RAISE = 2
  FOLD = 3

  def __init__(self, player_id, stack=2000, emptyplayer=False):
    self.player_id = player_id

    self.hand = []
    self.stack = stack
    self.hand_starting_stack = self.stack
    self.stack_for_street = self.stack
    self.currentbet = 0
    self.lastsidepot = 0
    self._seat = -1
    self.handrank = -1
    self.blind = 0

    # flags for table management
    self.emptyplayer = emptyplayer
    self.betting = False
    self.isallin = False
    self.playing_hand = False
    self.playedthisround = False
    self.sitting_out = False # not used at the moment, but might become useful in the future

  def get_seat(self):
    return self._seat

  def set_seat(self, value):
    self._seat = value

  def reset_hand(self):
    self._hand = []
    self.playedthisround = False
    self.betting = False
    self.isallin = False
    self.currentbet = 0
    self.lastsidepot = 0
    self.hand_starting_stack = self.stack
    self.stack_for_street = self.stack
    self.playing_hand = (self.stack != 0)

  def declare_action(self, bet_size):
    self.playedthisround = True
    if not bet_size:
      return
    self.stack -= (bet_size - self.currentbet)
    self.currentbet = bet_size
    self.stack_for_street = self.stack + self.currentbet
    if self.stack == 0:
      self.isallin = True

  def refund(self, ammount):
    self.stack += ammount

  def player_state(self):
    return (self.get_seat(), self.stack, self.playing_hand, self.betting, self.player_id)

  def reset_stack(self, amount=2000):
    self.stack = amount

  def update_localstate(self, table_state):
    self.stack = table_state.get('stack')
    self.hand = table_state.get('pocket_cards')

  # cleanup
  def validate_action(self, table_state, action):
    self.update_localstate(table_state)
    tocall = min(table_state.get('tocall', 0), self.stack_for_street)
    minraise = table_state.get('minraise', 0)

    [action_idx, raise_amount] = action
    raise_amount = int(raise_amount)
    action_idx = int(action_idx)

    if tocall - self.currentbet == 0:
      assert action_idx in [Player.CHECK, Player.RAISE]
      if action_idx == Player.RAISE:
        if raise_amount < minraise:
          raise error.Error('raise must be greater than minraise {}'.format(minraise))
        if raise_amount > self.stack_for_street:
          raise error.Error('raise must be less than maxraise {}'.format(self.stack_for_street))
        move_tuple = ('raise', raise_amount)
      elif action_idx == Player.CHECK:
        move_tuple = ('check', 0)
      else:
        raise error.Error('invalid action ({}) must be check (0) or raise (2)'.format(action_idx))
    else:
      if action_idx not in [Player.RAISE, Player.CALL, Player.FOLD]:
        raise error.Error('invalid action ({}) must be raise (2), call (1), or fold (3)'.format(action_idx))
      if action_idx == Player.RAISE:
        if raise_amount < minraise:
          raise error.Error('raise must be greater than minraise {}'.format(minraise))
        if raise_amount > self.stack_for_street:
          raise error.Error('raise must be less than maxraise {}'.format(self.stack_for_street))
        move_tuple = ('raise', raise_amount)
      elif action_idx == Player.CALL:
        move_tuple = ('call', tocall)
      elif action_idx == Player.FOLD:
        move_tuple = ('fold', 0)
    return move_tuple
