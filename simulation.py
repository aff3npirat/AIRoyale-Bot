import random

import torch
import numpy as np

from bots.single_deck.bot import SingleDeckBot
from bots.single_deck.nn import QNet
from bots.single_deck.bot import NEXT_CARD_END
from constants import PRINCESS_HP, KING_HP, TOWER_HP_BOXES



FPS = 30


class CRSim:
    
    def __init__(self, blue_cards, red_cards, king_levels=None):
        self.delta_seconds = 1 / FPS
        self.time = 0
        self.frame = 0

        self.king_levels = king_levels if king_levels is not None else {"blue": 11, "red": 11}
        self.overtime = False
        self.turret_hp = {}
        for name, _ in TOWER_HP_BOXES:
            level = self.king_levels["blue" if "ally" in name else "red"]

            if "princess" in name:
                hp = PRINCESS_HP[level]
            else:
                hp = KING_HP[level]

            self.turret_hp[name.replace("ally", "blue").replace("enemy", "red")] = hp

        self.blue_cards = sorted(blue_cards, key=lambda x: random.random())
        self.red_cards = sorted(red_cards, key=lambda x: random.random())

        self.blue_units = []  # (label, tile_x, tile_y)
        self.red_units = []

        # positions of tile next to bridge, used as target for units moving towards bridge
        self.left_path_tile_x = ...
        self.blue_path_tile_y = ...
        self.right_path_tile_x = ...
        self.red_path_tile_y = ...

        # height of king towers, used for units on/after bridge
        self.red_king_y = ...
        self.blue_king_y = ...

        # x coord, spliting board into left and right. All units with x<=left_side_x are left side.
        self.left_side_x = ...

        self.blue_units = np.empty((0, 11))  # label, x, y, health, action_time, attk_range, attk_speed, damage, sight_range, x_coeff, movement_speed
        self.red_units = np.empty((0, 11))


    def get_states(self):
        # TODO
        pass

    def update(self, actions):
        # TODO move, attack, repeat
        dists = np.abs(self.blue_units[:, (1, 2), None] - self.red_units[:, (1, 2), None].T)
        dists = np.sqrt(np.sum(dists**2, axis=1))
        
        # mask out any illegal targets (such as self, or flying)
        # TODO
        np.fill_diagonal(dists, np.inf)

        dists_blue = np.min(dists, axis=2)  # dist for each blue unit to closest red unit
        targets_blue = np.argmin(dists, axis=2)

        dists_red = np.min(dists, axis=1)
        targets_red = np.argmin(dists, axsis=1)

        n_blue = len(self.blue_units)
        n_red = len(self.red_units)

        units = np.concatenate((self.blue_units, self.red_units), axis=0)
        targets_idx = np.concatenate((targets_blue, targets_red + n_blue), axis=0)
        dists = np.concatenate((dists_blue, dists_red), axis=0)

        units[:, 4] += self.delta_seconds
        
        # attack unit
        mask_attack = (dists<=units[:, 5] & units[:, 4]>=0.0)
        if mask_attack.any():
            units[mask_attack, 4] -= units[mask_attack, 6]  # update action time
            units[targets_idx[mask_attack], 3] -= units[mask_attack, 7]  # update health

        # move units that did not attack
        target_tiles = np.zeros((n_blue + n_red, 2))

        for start, end, units_, y, king_y in (
            (0, n_blue, self.blue_units, self.blue_path_tile_y, self.red_king_y),
            (n_blue, n_blue+n_red, self.red_path_tile_y, self.blue_king_y),
        ):
            target_tiles[start:end, 1][units_[:, 2]>y] = y
            target_tiles[start:end, 1][units_[:, 2]<=y] = king_y
            target_tiles[start:end, 0][units_[:, 1]>self.left_side_x] = self.right_path_tile_x
            target_tiles[start:end, 0][units_[:, 1]<=self.left_side_x] = self.left_path_tile_x

        directions = target_tiles - units[:, (1, 2)]
        # directions[:, 0] *= units[:, 9]

        # override for units that have a target
        mask_sight = (dists<=units[8])
        directions[mask_sight & ~mask_attack] = units[targets_idx[mask_sight & ~mask_attack], (1, 2)] - units[mask_sight & ~mask_attack, (1, 2)]

        directions /= np.sqrt(np.sum(directions**2, axis=1))  # normalize to unit length
        directions *= self.units[:, 10] * self.delta_seconds # get movement in one frame

        units[:, (1, 2)] += directions

        self.blue_units, self.red_units = units[:n_blue], units[n_blue:]
