import os

import numpy as np
from PIL import Image

from screen import Screen
from constants import SCREEN_CONFIGS, SCREEN_HASH_THRES, DATA_DIR



class BotBase:

    def __init__(self, hash_size=8):
        self.screen = Screen()

        self.screen_hashes = {}
        self.hash_size = hash_size

        with Image.open(os.path.join(DATA_DIR, "images/screens/in_game.png"), mode="r") as I:
            ingame_hash = I.resize((hash_size, hash_size), Image.Resampling.BILINEAR).convert("L")
            ingame_hash = np.array(ingame_hash).flatten()
        self.screen_hashes["in_game"] = ingame_hash

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
        bbox = SCREEN_CONFIGS["in_game"]
        actual_hash = image.crop(bbox).resize((self.hash_size, self.hash_size), Image.Resampling.BILINEAR).convert("L")
        actual_hash = np.array(actual_hash).flatten()

        diff = np.mean(np.abs(self.screen_hashes["in_game"] - actual_hash))

        return diff < SCREEN_HASH_THRES
    
    @staticmethod
    def is_victory(image):
        """
        Returns True if image is victory screen.
        """
        raise NotImplementedError
    
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
