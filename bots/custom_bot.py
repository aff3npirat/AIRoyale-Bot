import os

import numpy as np
from PIL import Image

from screen import Screen
from constants import SCREEN_CONFIG, CONVERSION_MATS, DATA_DIR, CARD_HEIGHT, CARD_WIDTH, CARD_CONFIG



class BotBase:

    def __init__(self, hash_size=8, port=5555):
        self.screen = Screen(port=port)

        self.replay_buffer = []
        self.screen_hashes = {}
        self.hash_size = hash_size

        for key in SCREEN_CONFIG:
            mat = None
            if key in CONVERSION_MATS:
                mat = CONVERSION_MATS[key]
            with Image.open(os.path.join(DATA_DIR, f"images/screens/{key}.png"), mode="r") as I:
                self.screen_hashes[key] = BotBase._compute_image_hash(I, self.hash_size, conversion_mat=mat)

    @staticmethod
    def _compute_image_hash(image, hash_size, conversion_mat=None):
        mode = "L" if conversion_mat is None else None
        image_hash = image.resize((hash_size, hash_size), Image.Resampling.BILINEAR).convert(mode=mode, matrix=conversion_mat)
        image_hash = np.array(image_hash, dtype=float).flatten()
        return image_hash
    
    @staticmethod
    def slot_to_xy(slot):
        x, y = CARD_CONFIG[slot+1][:2]
        x = x + CARD_WIDTH/2
        y = y + CARD_HEIGHT/2

        return x, y
    
    @staticmethod
    def get_reward(*args, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def with_reward(episode, victory):
        """
        Returns triplets of <state, action, reward>
        """
        raise NotImplementedError
    
    def store_experience(self, state, actions):
        """
        Stores state, actions pair in replay buffer.
        """
        raise NotImplementedError

    def get_state(self, image):
        """
        Extracts state from image.
        """
        raise NotImplementedError
    
    def get_actions(self, state, eps):
        """
        Choose actions for given state.
        """
        raise NotImplementedError
    
    def play_actions(self, actions):
        """
        Execute given actions.
        """
        raise NotImplementedError
    
    def detect_game_screen(self, image, screen_key, conversion_mat=None):
        bbox, thr = SCREEN_CONFIG[screen_key]
        actual_hash = self._compute_image_hash(image.crop(bbox), self.hash_size, conversion_mat=conversion_mat)

        diff = np.mean(np.abs(self.screen_hashes[screen_key] - actual_hash))

        return diff < thr
    
    def is_game_end(self, image):
        """
        Returns True when on victory/loss screen.
        """
        return self.detect_game_screen(image, "game_end")
    
    def in_game(self, image):
        """
        Returns True when not in game.
        """
        conversion_matrix = np.zeros((4, 4))
        conversion_matrix[2, 2] = 1.0
        return self.detect_game_screen(image, "in_game", conversion_mat=conversion_matrix)
    
    def is_victory(self, image):
        """
        Returns True if image is victory screen.
        """
        bbox, thr = SCREEN_CONFIG["victory"]
        crop = image.crop(bbox).convert("L")
        
        crop = np.array(crop)
        mask = (crop<100)
        crop[mask] = 0
        crop[~mask] = 255

        actual_hash = self._compute_image_hash(Image.fromarray(crop), self.hash_size)
        diff = np.mean(np.abs(self.screen_hashes["victory"] - actual_hash))

        return diff < thr
    
    def run(self, eps):
        image = self.screen.take_screenshot()
        while not self.in_game(image):

            state = self.get_state(image)
            actions = self.get_actions(state, eps=eps)
            self.play_actions(actions)

            image = self.screen.take_screenshot()

        victory = self.is_victory(image)

        return victory
