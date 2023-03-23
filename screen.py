import subprocess

from PIL import Image

from timing import exec_time
from constants import SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT



class Screen:
    
    def __init__(self, adb_path=r"..\adb\adb.exe", port=5555):
        # Physical size: 720x1280 -> self.width = 720, self.height = 1280
        self.adb_exec = adb_path
        self.target = f"localhost:{port}"
        self.port = port
        window_size = subprocess.check_output(f"{self.adb_exec} -s {self.target} shell wm size")
        window_size = window_size.decode('ascii').replace('Physical size: ', '')
        self.width, self.height = [int(i) for i in window_size.split('x')]

    @exec_time
    def click(self, x, y):
        """
        Click at the given (x, y) coordinate
        """
        x = x / SCREENSHOT_WIDTH * self.width
        y = y / SCREENSHOT_HEIGHT * self.height

        subprocess.run(f"{self.adb_exec} -s {self.target} shell input tap {x} {y}")

    @exec_time
    def select_place_unit(self, slot_idx, side):
        select = f"dd bs=160 if=/mnt/sdcard/slot{slot_idx+1} of=/dev/input/event5"
        place = f"dd bs=160 if=/mnt/sdcard/tile_{side} of=/dev/input/event5"
        subprocess.run(rf"{self.adb_exec} -s {self.target} shell {select}; {place}", stderr=subprocess.DEVNULL)

    @exec_time
    def take_screenshot(self):
        """
        Take a screenshot of the emulator
        """
        screenshot_bytes = subprocess.run(f"{self.adb_exec} -s {self.target} exec-out screencap", check=True, capture_output=True).stdout
        screenshot = Image.frombuffer('RGBA', (self.width, self.height), screenshot_bytes[12:], 'raw', 'RGBX', 0, 1)
        screenshot = screenshot.convert('RGB').resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.BILINEAR)
        return screenshot


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cls = Screen()
    screenshot = cls.take_screenshot()
    plt.imshow(screenshot)
    plt.show()
