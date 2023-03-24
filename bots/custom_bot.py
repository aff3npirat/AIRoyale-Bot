import os

import numpy as np
from PIL import Image

from emulator import Controller
from timing import exec_time
from utils import compute_image_hash
from constants import SCREEN_CONFIG, DATA_DIR, CARD_HEIGHT, CARD_WIDTH, CARD_CONFIG



class BotBase:

    def __init__(self, hash_size=8, port=5555):
        self.controller = Controller(port=port)

        self.replay_buffer = []
        self.screen_hashes = {}
        self.hash_size = hash_size

        for key in SCREEN_CONFIG:
            with Image.open(os.path.join(DATA_DIR, f"images/screens/{key}.png"), mode="r") as I:
                if key == "in_game":
                    I = Image.fromarray(np.array(I)[..., 0])

                self.screen_hashes[key] = compute_image_hash(I, self.hash_size)
    
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
    
    @exec_time
    def detect_game_screen(self, image, screen_key):
        bbox, thr = SCREEN_CONFIG[screen_key]
        actual_hash = compute_image_hash(image.crop(bbox), self.hash_size)

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
        image = Image.fromarray(np.array(image)[..., 0])
        return self.detect_game_screen(image, "in_game")
    
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

        actual_hash = compute_image_hash(Image.fromarray(crop), self.hash_size)
        diff = np.mean(np.abs(self.screen_hashes["victory"] - actual_hash))

        return diff < thr
    
    def run(self, *args, **kwargs):
        raise NotImplementedError
