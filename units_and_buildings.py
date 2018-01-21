from pysc2.lib import actions
from pysc2.lib import features
import sparse_agent as sa
import random
"""
Complete list of IDs
https://pastebin.com/KCwwLiQ1
"""
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index


BUILD_COMMAND_CENTER = actions.FUNCTIONS.Build_CommandCenter_screen.id
BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
BUILD_STARPORT = actions.FUNCTIONS.Build_Starport_screen.id
BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
BUILD_FACTORY = actions.FUNCTIONS.Build_Factory_screen.id

TERRAN_COMMAND_CENTER = 18
TERRAN_SUPPLY_DEPOT = 19
TERRAN_REFINERY = 20
TERRAN_BARRACKS = 21
TERRAN_FACTORY = 27
TERRAN_STARPORT = 28

NEUTRAL_VESPENE_GAS = 342
NEUTRAL_MINERAL_FIELD = 341

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_COMMAND_CENTER = 'buildcommandcenter'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_STARPORT = 'buildstarport'
ACTION_BUILD_FACTORY = 'buildfactory'
ACTION_BUILD_REFINERY = 'buildrefinery'

BIG_PIX = 137
SMALL_PIX = 6

SMALL_GRID = 7
BIG_GRID = 12

COLUMN_HEIGHT = 5
SHARED_COLUMN = {BUILD_STARPORT, BUILD_BARRACKS, BUILD_FACTORY}


class Building:
    def __init__(self, name, identifier, build, smart_action, max_amount, size_grid, size_pixels):
        self.name = name
        self.identifier = identifier
        self.build = build
        self.smart_action = smart_action
        self.max_amount = max_amount
        self.size_grid = size_grid
        self.size_pixels = size_pixels

    @staticmethod
    def buildings():
        return [Building("Supply Depot", TERRAN_SUPPLY_DEPOT, BUILD_SUPPLY_DEPOT, ACTION_BUILD_SUPPLY_DEPOT, 5, SMALL_GRID, SMALL_PIX),
                Building("Barracks", TERRAN_BARRACKS, BUILD_BARRACKS, ACTION_BUILD_BARRACKS, 4, BIG_GRID, BIG_PIX),
                Building("Factory", TERRAN_FACTORY, BUILD_FACTORY, ACTION_BUILD_FACTORY, 2, BIG_GRID, BIG_PIX),
                Building("Starport", TERRAN_STARPORT, BUILD_STARPORT, ACTION_BUILD_STARPORT, 2, BIG_GRID, BIG_PIX),
                Building("Refinery", TERRAN_REFINERY, BUILD_REFINERY, ACTION_BUILD_REFINERY, 2, BIG_GRID, BIG_PIX)
               ]

    @staticmethod
    def get_location_from_id(_id, obs):
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _id).nonzero()
        if unit_y.any():
            i = random.randint(0, len(unit_y) - 1)
            m_x = unit_x[i]
            m_y = unit_y[i]
            return [m_x, m_y]

    def amount_of_building(self, obs):
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        _y, _ = (unit_type == self.identifier).nonzero()
        return int(round(len(_y) / self.size_pixels))

    def should_build(self, obs):
        return self.amount_of_building(obs) < self.max_amount and self.build in obs.observation['available_actions']

    def get_location(self, agent, obs):
        if self.build in SHARED_COLUMN:
            amount = sum(map(lambda b: b.amount_of_building(obs), filter(lambda b: b.build in SHARED_COLUMN, self.buildings())))
        elif self.identifier == TERRAN_REFINERY:
            return self.get_location_from_id(NEUTRAL_VESPENE_GAS, obs)
        elif self.identifier == TERRAN_SUPPLY_DEPOT:
            return agent.transformDistance(round(agent.cc_x.mean()), -35, round(agent.cc_y.mean()), -15 + 7 * self.amount_of_building(obs))
        else:
            amount = self.amount_of_building(obs)
        x_offset = 20 + int(amount/COLUMN_HEIGHT) * self.size_grid
        y_offset = -30 + self.size_grid * (amount % COLUMN_HEIGHT)
        return agent.transformDistance(round(agent.cc_x.mean()), x_offset, round(agent.cc_y.mean()), y_offset)

TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
TRAIN_MEDIVAC = actions.FUNCTIONS.Train_Medivac_quick.id

ACTION_TRAIN_MARINE = 'buildmarine'
ACTION_TRAIN_MEDIVAC = 'buildmedivac'
ACTION_TRAIN_SCV = 'buildscv'

TERRAN_SCV = 45
TERRAN_MARINE = 48
TERRAN_MEDIVAC = 54


class Unit:
    def __init__(self, name, identifier, build, smart_action, builds_from):
        self.name = name
        self.identifier = identifier
        self.build = build
        self.smart_action = smart_action
        self.builds_from = builds_from

    @staticmethod
    def units():
        return [Unit("SCV", TERRAN_SCV, TRAIN_SCV, ACTION_TRAIN_SCV, "Command Center"),
                Unit("Marine", TERRAN_MARINE, TRAIN_MARINE, ACTION_TRAIN_MARINE, "Barracks"),
                Unit("Medivac", TERRAN_MEDIVAC, TRAIN_MEDIVAC, ACTION_TRAIN_MEDIVAC, "Starport"),
               ]
