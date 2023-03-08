import subprocess

from PIL import Image

from constants import SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT



class Screen:
    
    def __init__(self, adb_path=r"..\adb\adb.exe"):
        # Physical size: 720x1280 -> self.width = 720, self.height = 1280
        self.adb_exec = adb_path
        window_size = subprocess.check_output([self.adb_exec, '-s', 'localhost:5555', 'shell', 'wm', 'size'])
        window_size = window_size.decode('ascii').replace('Physical size: ', '')
        self.width, self.height = [int(i) for i in window_size.split('x')]

    def click(self, x, y):
        """
        Click at the given (x, y) coordinate
        """
        x = x / SCREENSHOT_WIDTH * self.width
        y = y / SCREENSHOT_HEIGHT * self.height

        subprocess.run([self.adb_exec, '-s', 'localhost:5555', 'shell', 'input', 'tap', str(x), str(y)])

    def take_screenshot(self):
        """
        Take a screenshot of the emulator
        """
        screenshot_bytes = subprocess.run([self.adb_exec, '-s', 'localhost:5555', 'exec-out', 'screencap'], check=True, capture_output=True).stdout
        screenshot = Image.frombuffer('RGBA', (self.width, self.height), screenshot_bytes[12:], 'raw', 'RGBX', 0, 1)
        screenshot = screenshot.convert('RGB').resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.BILINEAR)
        return screenshot


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cls = Screen()
    screenshot = cls.take_screenshot()
    plt.imshow(screenshot)
    plt.show()
