import subprocess

from PIL import Image

from timing import exec_time
from constants import SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT, ADB_PATH



class Screen:
    
    def __init__(self, port=5555):
        # Physical size: 720x1280 -> self.width = 720, self.height = 1280
        self.target = f"localhost:{port}"
        self.port = port
        self.width, self.height = Screen.get_window_size(port)

    @staticmethod
    def get_window_size(port):
        window_size = subprocess.check_output(f"{ADB_PATH} -s localhost:{port} shell wm size")
        window_size = window_size.decode('ascii').replace('Physical size: ', '')
        width, height = [int(i) for i in window_size.split('x')]
        return width, height

    @exec_time
    def click(self, x, y):
        """
        Click at the given (x, y) coordinate
        """
        x = x / SCREENSHOT_WIDTH * self.width
        y = y / SCREENSHOT_HEIGHT * self.height

        subprocess.run(f"{ADB_PATH} -s {self.target} shell input tap {x} {y}")

    @exec_time
    def select_place_unit(self, slot_idx, side):
        select = f"dd bs=160 if=/mnt/sdcard/slot{slot_idx+1} of=/dev/input/event5"
        place = f"dd bs=160 if=/mnt/sdcard/tile_{side} of=/dev/input/event5"
        subprocess.run(rf"{ADB_PATH} -s {self.target} shell {select}; {place}", stderr=subprocess.DEVNULL)

    @exec_time
    def take_screenshot(self):
        """
        Take a screenshot of the emulator
        """
        screenshot_bytes = subprocess.run(f"{ADB_PATH} -s {self.target} exec-out screencap", check=True, capture_output=True).stdout
        screenshot = Image.frombuffer('RGBA', (self.width, self.height), screenshot_bytes[12:], 'raw', 'RGBX', 0, 1)
        screenshot = screenshot.convert('RGB').resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.BILINEAR)
        return screenshot


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cls = Screen()
    screenshot = cls.take_screenshot()
    plt.imshow(screenshot)
    plt.show()
