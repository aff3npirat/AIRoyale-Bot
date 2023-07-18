import os

import numpy as np
from PIL import Image

from utils import compute_image_hash
from timing import exec_time
from constants import SCREEN_CONFIG, DATA_DIR



class ScreenDetector:

    def __init__(self, hash_size):
        self.screen_hashes = {}
        self.hash_size = hash_size

        for key in SCREEN_CONFIG:
            img = Image.open(os.path.join(DATA_DIR, f"images/screens/{key}.png"), mode="r")
            if key == "in_game":
                img = Image.fromarray(np.array(img)[..., 0])

            self.screen_hashes[key] = compute_image_hash(img, self.hash_size)
            img.close()

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
