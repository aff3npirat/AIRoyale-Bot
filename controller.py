import subprocess
import time

from screen import Screen
from constants import (
    CLAN_MATCH,
    ADB_PATH,
    SCREENSHOT_WIDTH,
    SCREENSHOT_HEIGHT,
    CLAN_1V1,
    CLAN_1V1_ACCEPT,
    SCREEN_CONFIG
)



def accept_invite(screen):
    screen.click(*CLAN_1V1_ACCEPT)
    

def send_clan_1v1(port):
    width, height = Screen.get_window_size(port)

    x, y = CLAN_MATCH
    x, y = x / SCREENSHOT_WIDTH * width, y / SCREENSHOT_HEIGHT * height
    subprocess.run(f"{ADB_PATH} -s localhost:{port} shell input tap {x} {y}")
    time.sleep(1)
    for _ in range(5):
        subprocess.run(f"{ADB_PATH} -s localhost:{port} shell input swipe 150 550 150 150 200")
    time.sleep(1)

    x, y = CLAN_1V1
    x, y = x / SCREENSHOT_WIDTH * width, y / SCREENSHOT_HEIGHT * height
    subprocess.run(f"{ADB_PATH} -s localhost:{port} shell input tap {x} {y}")


def exit_game(screen):
    x1, y1, x2, y2 = SCREEN_CONFIG["game_end"][0]
    x, y = x1 + (x2-x1)/2, y1 + (y2-y1)/2

    screen.click(x, y)
