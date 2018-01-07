# -*- coding: utf-8 -*-
from gym.envs.registration import register

from .env import TexasHoldemEnv
from .utils import card_to_str, hand_to_str, safe_actions, action_table

register(
 	id='TexasHoldem-v0',
 	entry_point='holdem.env:TexasHoldemEnv',
  kwargs={'n_seats': 2, 'debug': False},
)

register(
 	id='TexasHoldem-v1',
 	entry_point='holdem.env:TexasHoldemEnv',
  kwargs={'n_seats': 4, 'debug': False},
)

register(
 	id='TexasHoldem-v2',
 	entry_point='holdem.env:TexasHoldemEnv',
  kwargs={'n_seats': 8, 'debug': False},
)
