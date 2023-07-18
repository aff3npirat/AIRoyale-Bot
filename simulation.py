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

        self.units = []  # label, x, y, health, action_time, attk_range, attk_speed, damage, sight_range

    def get_states(self):
        # TODO
        pass

    def update(self, actions):
        # TODO move, attack, repeat
        self.units[4] += self.delta_seconds

        dists = np.abs(self.units[:, (1, 2), None] - self.units[:, (1, 2), None].T)
        dists = np.sqrt(np.sum(dists**2, axis=1))
        np.fill_diagonal(dists, np.inf)

        target_dist = np.min(dists, axis=1)
        target_idx = np.argmin(dists, axis=1)

        # attack unit
        mask = (target_dist<=self.units[5])  # dist <= attk_range
        if mask.any():
            self.units[mask, 4] -= self.units[mask, 6]  # update action time
            self.units[target_idx[mask], 3] -= self.units[mask, 7]  # update health
        




def run(eps, blue_cards, red_cards, king_levels=None):
    sim = CRSim(
        blue_cards=blue_cards, red_cards=red_cards, king_levels=king_levels
    )

    qnet = QNet([512+NEXT_CARD_END, 128, 64, 5], activation="sigmoid", bias=True, feature_extractor=False)


    while not sim.game_over:
        blue_state, red_state = sim.get_states()
        
        actions = {}
        for state, team in zip((blue_state, red_state), ("blue", "red")):
            illegal = SingleDeckBot.get_illegal_actions(state)

            if random.random() < eps:
                action = random.choice(list(range(5)))
                if action in illegal:
                    action = 0
            else:
                qvals = qnet(state)
                qvals[illegal] = -torch.inf

                action = torch.argmax(qvals)
        
            actions[team] = action

        sim.update(actions)
    

    



