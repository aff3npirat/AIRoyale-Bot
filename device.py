import os
import subprocess
import time

import numpy as np
from PIL import Image

from utils import compute_image_hash
from timing import exec_time
from constants import (
    CLAN_MATCH,
    ADB_PATH,
    SCREENSHOT_WIDTH,
    SCREENSHOT_HEIGHT,
    CLAN_1V1,
    CLAN_1V1_ACCEPT,
    SCREEN_CONFIG,
    DATA_DIR,
)



def get_window_size(device):
    window_size = subprocess.check_output(f"{ADB_PATH} -s {device} shell wm size")
    window_size = window_size.decode('ascii').replace('Physical size: ', '')
    width, height = [int(i) for i in window_size.split('x')]
    return width, height


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


class Controller:

    def __init__(self, device):
        self.device = device
        self.width, self.height = get_window_size(self.device)

    @exec_time
    def click(self, x, y):
        x = x / SCREENSHOT_WIDTH * self.width
        y = y / SCREENSHOT_HEIGHT * self.height

        subprocess.run(f"{ADB_PATH} -s {self.device} shell input tap {x} {y}")

    @exec_time
    def take_screenshot(self):
        screenshot_bytes = subprocess.run(f"{ADB_PATH} -s {self.device} exec-out screencap", check=True, capture_output=True).stdout
        screenshot = Image.frombuffer('RGBA', (self.width, self.height), screenshot_bytes[12:], 'raw', 'RGBX', 0, 1)
        screenshot = screenshot.convert('RGB').resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.BILINEAR)
        return screenshot

    def accept_invite(self):
        self.click(*CLAN_1V1_ACCEPT)

    def send_clan_1v1(self):
        x, y = CLAN_MATCH
        x, y = x / SCREENSHOT_WIDTH * self.width, y / SCREENSHOT_HEIGHT * self.height
        subprocess.run(f"{ADB_PATH} -s {self.device} shell input tap {x} {y}")
        time.sleep(1)
        for _ in range(5):
            subprocess.run(f"{ADB_PATH} -s {self.device} shell input swipe 150 550 150 150 200")
        time.sleep(1)

        self.click(*CLAN_1V1)

    def exit_game(self):
        x1, y1, x2, y2 = SCREEN_CONFIG["game_end"][0]
        x, y = x1 + (x2-x1)/2, y1 + (y2-y1)/2

        self.click(x, y)

    @exec_time
    def select_place_unit(self, slot_idx, tile):
        select = f"dd bs=160 if=/mnt/sdcard/slot{slot_idx+1} of=/dev/input/event5"
        wait = "busybox usleep 20000"
        place = f"dd bs=160 if=/mnt/sdcard/tile_{tile} of=/dev/input/event5"
        subprocess.run(rf"{ADB_PATH} -s {self.device} shell {select}; {wait}; {place}", stderr=subprocess.DEVNULL)

