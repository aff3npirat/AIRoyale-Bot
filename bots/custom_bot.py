import os

import numpy as np
from PIL import Image

from screen import Screen
from constants import SCREEN_CONFIGS, DATA_DIR



class BotBase:

    def __init__(self, hash_size=8):
        self.screen = Screen()

        self.screen_hashes = {}
        self.hash_size = hash_size

        with Image.open(os.path.join(DATA_DIR, "images/screens/in_game.png"), mode="r") as I:
            self.screen_hashes["in_game"] = self._compute_image_hash(I)

        self.screen_hashes["victory"] = np.load(os.path.join(DATA_DIR, "images/screens/victory_hash.npy"), allow_pickle=False)

    def _compute_image_hash(self, image):
        image_hash = image.resize((self.hash_size, self.hash_size), Image.Resampling.BILINEAR).convert("L")
        image_hash = np.array(image_hash).flatten()
        return image_hash

    def get_state(self, image):
        """
        Extracts state from image.
        """
        raise NotImplementedError
    
    def get_actions(self, state):
        """
        Choose actions for given state.
        """
        raise NotImplementedError
    
    def play_actions(self, actions):
        """
        Execute given actions.
        """
        raise NotImplementedError
    
    @staticmethod
    def detect_game_screen(image):
        """
        Detect what screen (in game, home, etc..) is visible.
        """
        raise NotImplementedError
    
    def is_game_end(self, image):
        """
        Returns True when not in game.
        """
        bbox, thr = SCREEN_CONFIGS["in_game"]
        actual_hash = self._compute_image_hash(image.crop(bbox))

        diff = np.mean(np.abs(self.screen_hashes["in_game"] - actual_hash))

        return diff < thr
    
    def is_victory(self, image):
        """
        Returns True if image is victory screen.
        """
        bbox, thr = SCREEN_CONFIGS["victory"]
        actual_hash = self._compute_image_hash(image.crop(bbox))

        diff = np.mean(np.abs(self.screen_hashes["victory"] - actual_hash))

        return diff < thr
    
    def run(self, auto_play):
        if auto_play:
            # navigate from current screen to in game
            raise NotImplementedError
        
        image = self.screen.take_screenshot()
        while not self.is_game_end(image):

            state = self.get_state(image)
            actions = self.get_actions(state)
            self.play_actions(actions)

            image = self.screen.take_screenshot()

        victory = self.is_victory(image)
