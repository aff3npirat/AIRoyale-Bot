import numpy as np

from utils import masked_argmin
from simulation.simulation import Simuation



class BaseUnit:

    def __init__(self, name, x, y, health, range, sight_range, speed, attack_speed, unit_damage, tower_damage):
        self.pos = np.array([x, y])
        self.name = name

        self.health = health
        self.range = range
        self.sight_range = sight_range
        self.speed = speed
        self.attack_speed = attack_speed
        self.unit_damage = unit_damage
        self.tower_damage = tower_damage

        self.last_target = None
        self.action_timer = 0

    def walk(self, board_directions):
        raise NotImplementedError
    
    def update(self, dists : np.array, targets, tower_dists : np.array, towers, delta, valid_pos, board_directions):
        raise NotImplementedError
    
    def is_alive(self):
        raise NotImplementedError


class GroundUnit(BaseUnit):

    def __init__(self, name, x, y, health, range, sight_range, speed, attack_speed, unit_damage, tower_damage):
        super().__init__(name, x, y, health, range, sight_range, speed, attack_speed, unit_damage, tower_damage)

    def walk(self, board_directions):
        """Moves unit along default path.
        
        Args:
            board_directions: Array width board same dimensions as board. Element at position (x, y)
                gives the direction of the default path.
        """
        x, y = Simuation.pos2tile(self.pos[0], self.pos[1])
        d = board_directions[x, y]

        self.pos += d * self.speed

    def update(self, dists : np.array, targets, tower_dists : np.array, towers, delta, valid_pos, board_directions):
        """Updates the position of and attack timer.
        
        Args:
            dists: Distances to each valid target on board.
            targets: All valid targets.
            tower_dists: Distances to each enemy tower.
            towers: All enemy towers.
            delta: Time passed by in seconds.
            valid_pos: A function that expects a 2-element numpy array and
                returns True if position is valid. False otherwise.
            board_directions: See `GroundUnit.walk`.
            
        Returns:
            target: Index of target attacked by this unit. If no target was attacked -1.
            target_damage: Damage done to `target`. None if no unit was attacked.
            tower: Index of tower attacked by this unit. If no tower was attacked -1.
            tower_dmg: Damage done to tower. None if no damage was done.
        """
        target, target_damage = -1, None
        tower, tower_damage = -1, None

        self.action_timer += delta

        if self.last_target is not None and np.sqrt(np.sum((self.last_target.pos - self.pos)**2)) <= self.range:  # attack last target
            target = targets.index(self.last_target)
            target_damage = self.unit_damage
        else:  # search for new target
            mask = (dists <= self.sight_range)
            if mask.any():  # at least one new target found
                t = masked_argmin(dists, mask)
                dist = dists[t]

                if dist <= self.range and self.action_timer >= 0:  # new target in attack range
                    target, target_damage = t, self.unit_damage
                    self.last_target = targets[t]
                else:  # walk to new target
                    d = targets[t].pos - self.pos
                    d /= np.linspace.norm(self.pos, targets[t].pos, ord=None)  # normalize to unit length
                    new_pos = self.pos + d*self.speed

                    if valid_pos(new_pos):
                        self.pos = new_pos
                    else:
                        self.walk(board_directions)
            else:  # no new target found, look for towers
                mask = (tower_dists <= self.range)
                if mask.any():  # at least one tower found that can be attacked
                    t = masked_argmin(tower_dists, mask)
                    tower, tower_damage = t, self.tower_damage
                    self.last_target = towers[t]
                else:  # walk closer to tower
                    self.walk(board_directions)

        if self.action_timer < 0:
            target, target_damage = -1, None
            tower, tower_damage = -1, None
        else:
            if target >= 0 or tower >= 0:
                self.action_timer -= self.attack_speed

        return target, target_damage, tower, tower_damage
    
    def is_alive(self):
        return self.health > 0.0
    

class AirUnit(GroundUnit):
    """Flying Unit.
    
    A flying unit is only different to a ground unit that it can be on any tile.
    """

    def update(self, dists: np.array, targets, tower_dists: np.array, towers, delta, valid_pos, board_directions):
        return super().update(dists, targets, tower_dists, towers, delta, lambda x: True, board_directions)


class Building(GroundUnit):
    """A non-moving unit."""

    def __init__(self, name, x, y, health, range, sight_range, speed, attack_speed, unit_damage, tower_damage):
        super().__init__(name, x, y, health, range, sight_range, speed, attack_speed, unit_damage, tower_damage)

        self._pos = np.array([x, y])
    
    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, new_pos):  # to use `GroundUnit.update` disable any value changes to `pos`.
        pass

    def walk(self, board_directions):
        pass





