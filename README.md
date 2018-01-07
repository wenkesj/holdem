# holdem

:warning: **This is an experimental API, it will most definitely contain bugs, but that's why you are here!**

```sh
pip install holdem
```

Afaik, this is the first [OpenAI Gym](https://github.com/openai/gym) _No-Limit Texas Hold'em_* (NLTH)
environment written in Python. It's an experiment to build a Gym environment that is synchronous and
can support any number of players but also appeal to the general public that wants to learn how to
"solve" NLTH.

*Python 3 supports arbitrary length integers :money_with_wings:

Right now, this is a work in progress, but I believe the API is mature enough for some preliminary
experiments. Join me in making some interesting progress on multi-agent Gym environments.

# Usage

There is limited documentation at the moment. I'll try to make this less painful to understand.

## `env = holdem.TexasHoldemEnv(n_seats, max_limit=1e9, debug=False)`

Creates a gym environment representation a NLTH Table from the parameters:

+ `n_seats` - number of available players for the current table. No players are initially allocated
  to the table. You must call `env.add_player(seat_id, ...)` to populate the table.
+ `max_limit` - max_limit is used to define the `gym.spaces` API for the class. It does not actually
  determine any NLTH limits; in support of `gym.spaces.Discrete`.
+ `debug` - add debug statements to play, will probably be removed in the future.

### `env.add_player(seat_id, stack=2000)`

Adds a player to the table according to the specified seat (`seat_id`) and the initial amount of
chips allocated to the player's `stack`. If the table does not have enough seats according to the
`n_seats` used by the constructor, a `gym.error.Error` will be raised.

### `(player_states, community_states) = env.reset()`

Calling `env.reset` resets the NLTH table to a new hand state. It does not reset any of the players
stacks, or, reset any of the blinds. New behavior is reserved for a special, future portion of the
API that is yet another feature that is not standard in Gym environments and is a work in progress.

The observation returned is a `tuple` of the following by index:

0. `player_states` - a `tuple` where each entry is `tuple(player_info, player_hand)`, this feature
   can be used to gather all states and hands by `(player_infos, player_hands) = zip(*player_states)`.
   + `player_infos` - is a `list` of `int` features describing the individual player. It contains
     the following by index:
     0. `[0, 1]` - `0` - seat is empty, `1` - seat is not empty.
     1. `[0, n_seats - 1]` - player's id, where they are sitting.
     2. `[0, inf]` - player's current stack.
     3. `[0, 1]` - player is playing the current hand.
     4. `[0, inf]` the player's current handrank according to `treys.Evaluator.evaluate(hand, community)`.
     5. `[0, 1]` - `0` - player has not played this round, `1` - player has played this round.
     6. `[0, 1]` - `0` - player is currently not betting, `1` - player is betting.
     7. `[0, 1]` - `0` - player is currently not all-in, `1` - player is all-in.
     8. `[0, inf]` - player's last sidepot.
   + `player_hands` - is a `list` of `int` features describing the cards in the player's pocket.
     The values are encoded based on the `treys.Card` integer representation.
1. `community_states` - a `tuple(community_infos, community_cards)` where:
   + `community_infos` - a `list` by index:
     0. `[0, n_seats - 1]` - location of the dealer button, where big blind is posted.
     1. `[0, inf]` - the current small blind amount.
     2. `[0, inf]` - the current big blind amount.
     3. `[0, inf]` - the current total amount in the community pot.
     4. `[0, inf]` - the last posted raise amount.
     5. `[0, inf]` - minimum required raise amount, if above 0.
     6. `[0, inf]` - the amount required to call.
     7. `[0, n_seats - 1]` - the current player required to take an action.
   + `community_cards` - is a `list` of `int` features describing the cards in the community.
     The values are encoded based on the `treys.Card` integer representation. There are 5 `int` in
     the list, where `-1` represents that there is no card present.

# Example

```python
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
```
