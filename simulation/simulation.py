import copy

import numpy as np



class Simuation:

    def __init__(self, blue_deck, red_deck):
        """Initializes a simulated game of CR.
        
        Args:
            blue_deck, red_deck: List of card labels.
        """
        self.blue_units = []
        self.red_units = []

        self.blue_towers = []
        self.red_towers = []
        self.tower_dims = np.array(...)
        self.blue_tower_mask = np.array([True, True, True])
        self.red_tower_mask = np.array([True, True, True])
        self.blue_king, self.red_king = self.blue_towers[1], self.red_towers[1]

        self.blue_board_directions = np.array(...)
        self.red_board_directions = np.array(...)

    @staticmethod
    def get_card_stats(path, name):
        # TODO
        raise NotImplementedError

    @staticmethod
    def pos2tile(x, y):
        # TODO
        raise NotImplementedError

    def valid_pos(self, pos):
        # TODO
        raise NotImplementedError

    @staticmethod
    def tower_dists(unit_pos, tower_pos, tower_dims):
        min_pos, max_pos = tower_pos - tower_dims/2, tower_pos + tower_dims/2

        dx = np.maximum(
            np.maximum(
                min_pos[:, 0, None].T - unit_pos[:, 0, None], unit_pos[:, 0, None] - max_pos[:, 0, None].T
            ), 0)
        
        dy = np.maximum(
            np.maximum(
                min_pos[:, 1, None].T - unit_pos[:, 1, None], unit_pos[:, 1, None] - max_pos[:, 1, None].T
            ), 0)
        
        return np.sqrt(dx**2 + dy**2)

    def update(self):
        """Updates the game for a fixed number of seconds.
        
        Returns:
            game_over: True if game is over.
            victory: None if game is not over. Otherwise one of {red, blue, tie}.
        """
        blue_pos = np.array([unit.pos for unit in self.blue_units])
        red_pos = np.array([unit.pos for unit in self.red_units])
        dists = np.sqrt(np.sum((blue_pos[:, :, None] - red_pos[:, :, None].T)**2, axis=1))

        blue_tower_pos = np.array([tower.pos for tower in self.blue_towers])
        red_tower_pos = np.array([tower.pos for tower in self.red_towers])
        # compute distance from each unit to each enemy tower.
        # `blue_tower_dists` are distances from red units to blue towers.
        blue_tower_dists = Simuation.tower_dists(red_pos, blue_tower_pos, self.tower_dims[self.blue_tower_mask])
        red_tower_dists = Simuation.tower_dists(blue_pos, red_tower_pos, self.tower_dims[self.red_tower_mask])

        n_blue = len(self.blue_units)
        blue_units = copy.deepcopy(self.blue_units)
        red_units = copy.deepcopy(self.red_units)
        for i, unit in enumerate(self.blue_units + self.red_units):
            if i < n_blue:
                dists_ = dists[i]
                targets = red_units
                tower_dists = red_tower_dists
                towers = self.red_towers
                board_dirs = self.blue_board_directions

                # which units need health updates
                update_units = self.red_units
                update_towers = self.red_towers
            else:
                dists_ = dists[:, i]
                targets = blue_units
                tower_dists = blue_tower_dists
                towers = self.blue_towers
                board_dirs = self.red_board_directions

                update_units = self.blue_units
                update_towers = self.blue_towers

            target, target_dmg, tower, tower_dmg = unit.update(
                dists=dists_, targets=targets, tower_dists=tower_dists, towers=towers,
                delta=self.delta, valid_pos=self.valid_pos, board_directions=board_dirs
            )

            if target >= 0:
                update_units[target].health -= target_dmg
            if tower >= 0:
                update_towers[tower].health -= tower_dmg

        for units in [self.blue_units, self.red_units]:
            for i, unit in enumerate(units):
                if unit.is_alive():
                    if not unit.last_target.is_alive():
                        unit.last_target = None
                else:
                    del units[i]

        for towers, mask in [(self.blue_towers, self.blue_tower_mask), (self.red_towers, self.red_tower_mask)]:
            for i, tower in enumerate(towers):
                if tower.is_alive():
                    if not tower.last_target.is_alive():
                        tower.last_target = None
                else:
                    mask[mask.nonzero()[0][i]] = False
                    del towers[i]

        if not all(self.blue_tower_mask[::2]):
            self.blue_king.action_timer = 0.0
        if not all(self.red_tower_mask[::2]):
            self.red_king.action_timer = 0.0

        game_over = not (self.blue_tower_mask[1] and self.red_tower_mask[1])
        victory = None
        if game_over:
            if self.blue_tower_mask[1]:
                victory = "blue"
            elif self.red_tower_mask[1]:
                victory = "red"
            else:
                victory = "tie"

        return game_over, victory
    
    def run(self, agent_blue, agent_red=None):
        """Play a single game from start to end.
        
        Args:
            agent_blue, agent_red: Callable that accepts a game state and returns an action.
            
        Returns:
            states: All visited states.
            actions: All actions performed.
            victory: Either "red", "blue", or "tie".
        """
        # TODO
        raise NotImplementedError
    
    def next_state(self, state, action_blue, action_red):
        """Play a single frame.

        Starts from a state, performs one action per team and updates state.

        Args:
            state: State to start from.
            action_blue, action_red: Actions to perform.

        Returns:
            next_state: The next state.
        """
        # TODO
        raise NotImplementedError
