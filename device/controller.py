import subprocess
import time

from PIL import Image

from timing import exec_time
from constants import (
    CLAN_MATCH,
    ADB_PATH,
    SCREENSHOT_WIDTH,
    SCREENSHOT_HEIGHT,
    CLAN_1V1,
    CLAN_1V1_ACCEPT,
    SCREEN_CONFIG,
    CARD_CONFIG,
)



def get_window_size(device):
    window_size = subprocess.check_output(f"{ADB_PATH} -s {device} shell wm size")
    window_size = window_size.decode('ascii').replace('Physical size: ', '')
    width, height = [int(i) for i in window_size.split('x')]
    return width, height


class Controller:

    def __init__(self, device):
        self.device = device
        self.width, self.height = get_window_size(self.device)

    @staticmethod
    def slot_to_xy(slot_idx):
        x1, y1, x2, y2 = CARD_CONFIG[slot_idx+1]
        x = x1 + (x2 - x1)/2
        y = y1 + (y2 - y1)/2

        return x, y

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
        select = f"dd bs=160 if=/sdcard/slot{slot_idx+1} of=/dev/input/event5"
        wait = "busybox usleep 20000"
        place = f"dd bs=160 if=/sdcard/tile_{tile} of=/dev/input/event5"
        subprocess.run(rf"{ADB_PATH} -s {self.device} shell {select}; {wait}; {place}", stderr=subprocess.DEVNULL)

