import random
import math
import os

import numpy as np
import pandas as pd

from random import randint
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_STARPORT = actions.FUNCTIONS.Build_Starport_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_FACTORY = actions.FUNCTIONS.Build_Factory_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_MEDIVAC = actions.FUNCTIONS.Train_Medivac_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_SHARED_COLUMN = {_BUILD_STARPORT, _BUILD_BARRACKS, _BUILD_FACTORY}

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_REFINERY = 20
_TERRAN_BARRACKS = 21
_TERRAN_FACTORY = 27
_TERRAN_STARPORT = 28
_TERRAN_SCV = 45
_NEUTRAL_MINERAL_FIELD = 341
_NEUTRAL_VESPENE_GAS = 342

_IDS = {_BUILD_BARRACKS: _TERRAN_BARRACKS,
       _BUILD_STARPORT: _TERRAN_STARPORT,
       _BUILD_FACTORY: _TERRAN_FACTORY,
       _BUILD_REFINERY: _TERRAN_REFINERY,
       _BUILD_SUPPLY_DEPOT: _TERRAN_SUPPLY_DEPOT,
 }

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'sparse_agent_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_TRAIN_MARINE = 'buildmarine'
ACTION_TRAIN_MEDIVAC = 'buildmedivac'
ACTION_BUILD_STARPORT = 'buildstarport'
ACTION_BUILD_FACTORY = 'buildfactory'
ACTION_BUILD_REFINERY = 'buildrefinery'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_TRAIN_MARINE,
    ACTION_TRAIN_MEDIVAC,
    ACTION_BUILD_STARPORT,
    ACTION_BUILD_FACTORY,
    ACTION_BUILD_REFINERY,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' +
                                 str(mm_x - 16) + '_' + str(mm_y - 16))

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class SparseAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_action = None
        self.previous_state = None
        self.cc_y = None
        self.cc_x = None
        self.move_number = 0

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def select_workers(self, unit_type):
        unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
        if unit_y.any():
            i = random.randint(0, len(unit_y) - 1)
            target = [unit_x[i], unit_y[i]]
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

    def unit_attack(self, x, y, obs):
        if self.move_number == 0:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        elif self.move_number == 1:
            do_it = True

            if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                do_it = False

            if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
                do_it = False

            if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                x_offset = random.randint(-1, 1)
                y_offset = random.randint(-1, 1)
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])

    def train_unit(self, unit_type, building_type, obs):
        if self.move_number == 0:
            target = self.get_location(building_type, obs)
            if target is not None:
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
        if self.move_number == 1:
            if unit_type in obs.observation['available_actions']:
                return actions.FunctionCall(unit_type, [_QUEUED])

    def build_supply_depot(self, count, obs):
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        if self.move_number == 0:
            return self.select_workers(unit_type)
        elif self.move_number == 1:
            if count < 10 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                if self.cc_y.any():
                    target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -15 + 7 * count)
                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
        elif self.move_number == 2:
            if _HARVEST_GATHER in obs.observation['available_actions']:
                return self.return_worker_to_harvest(obs)

    # assumes a worker is selected
    def return_worker_to_harvest(self, obs):
        r = randint(0, 21)
        target = self.get_location(_NEUTRAL_VESPENE_GAS, obs) if r < 6 else \
            self.get_location(_NEUTRAL_MINERAL_FIELD, obs)
        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])

    def build_target(self, obs, building_type, target, max_amount):
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        if self.move_number == 0:
            return self.select_workers(unit_type)
        elif self.move_number == 1:
            amount = self.amount_of_building(_IDS[building_type], unit_type)
            if amount < max_amount and building_type in obs.observation['available_actions']:
                return actions.FunctionCall(building_type, [_NOT_QUEUED, target])
        elif self.move_number == 2:
            if _HARVEST_GATHER in obs.observation['available_actions'] and building_type is not _BUILD_REFINERY:
                return self.return_worker_to_harvest(obs)

    def build(self, obs, building_type, max_amount):
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        if building_type in _SHARED_COLUMN:
            amount = sum(map(lambda b: self.amount_of_building(_IDS[b], unit_type), _SHARED_COLUMN))
        else:
            amount = self.amount_of_building(_IDS[building_type], unit_type)
        target = self.transformDistance(round(self.cc_x.mean()), 30, round(self.cc_y.mean()), -30 + 11.5 * amount)
        return self.build_target(obs, building_type, target, max_amount)

    @staticmethod
    def get_location(_id, obs):
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _id).nonzero()
        if unit_y.any():
            i = random.randint(0, len(unit_y) - 1)
            m_x = unit_x[i]
            m_y = unit_y[i]
            return [m_x, m_y]

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]
        return [x, y]

    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]
        y = x = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    def amount_of_building(self, building_id, unit_type):
        sizes = {_TERRAN_BARRACKS: 137,
                 _TERRAN_STARPORT: 137,
                 _TERRAN_FACTORY: 137,
                 _TERRAN_REFINERY: 137,
                 _TERRAN_SUPPLY_DEPOT: 69,
                 }
        _y, _ = (unit_type == building_id).nonzero()
        return int(round(len(_y) / sizes[building_id]))

    def step(self, obs):
        super(SparseAgent, self).step(obs)

        if obs.last():
            reward = obs.reward

            self.qlearn.learn(str(self.previous_state),
                              self.previous_action, reward, 'terminal')

            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
            self.previous_action = None
            self.previous_state = None
            self.move_number = 0

            return actions.FunctionCall(_NO_OP, [])

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        if obs.first():
            player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
            self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0

        supply_depot_count = self.amount_of_building(_TERRAN_SUPPLY_DEPOT, unit_type)
        building_types = [_TERRAN_BARRACKS, _TERRAN_STARPORT, _TERRAN_FACTORY]
        building_count = sum(map(lambda b: self.amount_of_building(b, unit_type), building_types))

        if self.move_number == 0:
            current_state = np.zeros(8)
            current_state[0] = cc_count
            current_state[1] = supply_depot_count
            current_state[2] = building_count
            current_state[3] = obs.observation['player'][_ARMY_SUPPLY]

            hot_squares = np.zeros(4)
            enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))

                hot_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                hot_squares = hot_squares[::-1]

            for i in range(0, 4):
                current_state[i + 4] = hot_squares[i]

            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state),
                                  self.previous_action, 0, str(current_state))

            rl_action = self.qlearn.choose_action(str(current_state))

            self.previous_state = current_state
            self.previous_action = rl_action

        smart_action, x, y = self.splitAction(self.previous_action)

        if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            move = self.build_supply_depot(supply_depot_count, obs)
        elif smart_action == ACTION_BUILD_BARRACKS:
            move = self.build(obs, _BUILD_BARRACKS, 2)
        elif smart_action == ACTION_BUILD_REFINERY:
            target = self.get_location(_NEUTRAL_VESPENE_GAS, obs)
            move = self.build_target(obs, _BUILD_REFINERY, target, 2)
        elif smart_action == ACTION_BUILD_STARPORT:
            move = self.build(obs, _BUILD_STARPORT, 1)
        elif smart_action == ACTION_BUILD_FACTORY:
            move = self.build(obs, _BUILD_FACTORY, 1)
        elif smart_action == ACTION_TRAIN_MARINE:
            move = self.train_unit(_TRAIN_MARINE, _TERRAN_BARRACKS, obs)
        elif smart_action == ACTION_TRAIN_MEDIVAC:
            move = self.train_unit(_TRAIN_MEDIVAC, _TERRAN_STARPORT, obs)
        elif smart_action == ACTION_ATTACK:
            move = self.unit_attack(x, y, obs)
        else:
            move = actions.FunctionCall(_NO_OP, [])

        if move is None:
            move = actions.FunctionCall(_NO_OP, [])
        
        self.move_number = 0 if self.move_number == 3 else self.move_number + 1
        return move
