# -*- coding: utf-8 -*-
#
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
import gym
import holdem

def play_out_hand(env, n_seats):
  # reset environment, gather relevant observations
  (player_states, (community_infos, community_cards)) = env.reset()
  (player_infos, player_hands) = zip(*player_states)

  # display the table, cards and all
  env.render(mode='human')

  terminal = False
  while not terminal:
    # play safe actions, check when noone else has raised, call when raised.
    actions = holdem.safe_actions(community_infos, n_seats=n_seats)
    (player_states, (community_infos, community_cards)), rews, terminal, info = env.step(actions)
    env.render(mode='human')

env = gym.make('TexasHoldem-v1') # holdem.TexasHoldemEnv(2)

# start with 2 players
env.add_player(0, stack=2000) # add a player to seat 0 with 2000 "chips"
env.add_player(1, stack=2000) # add another player to seat 1 with 2000 "chips"
# play out a hand
play_out_hand(env, env.n_seats)

# add one more player
env.add_player(2, stack=2000) # add another player to seat 1 with 2000 "chips"
# play out another hand
play_out_hand(env, env.n_seats)
