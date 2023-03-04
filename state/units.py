import numpy as np
from PIL import Image

from state.onnx_detector import OnnxDetector
from constants import (
    SIDE_H,
    SIDE_W,
    UNIT_H,
    UNIT_W,
    UNIT_NAMES,
    CARD_TO_UNITS,
    TILE_INIT_X,
    TILE_INIT_Y,
    TILE_WIDTH,
    TILE_HEIGHT,
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
    CARD_Y
)



class UnitDetector(OnnxDetector):
    
    def __init__(self, model_path, side_detector_path, ally_cards):
        super().__init__(model_path)
        self.side_detector = SideDetector(side_detector_path)

        ally_units = []
        for card in ally_cards:
            if card in CARD_TO_UNITS:
                units = CARD_TO_UNITS[card]
                if not isinstance(units, list):
                    units = [units]

                ally_units += units
            else:
                ally_units.append(card)
                    
        self.ally_units = ally_units

    @staticmethod
    def preprocess(img):
        img = img.resize((UNIT_H, UNIT_W), Image.BICUBIC)
        img = np.array(img, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img / 255
    
    def calculate_side(self, image, predictions):
        bboxes = np.round(predictions[:, :4])
        sides = np.ones(len(predictions))
        for i in range(len(predictions)):
            l = int(predictions[i, 5])

            name = UNIT_NAMES[l]
            if name not in self.ally_units:
                continue
            else:
                crop = image.crop(bboxes[i])
                sides[i] = self.side_detector.run(crop)

        return sides
    
    @staticmethod
    def box_to_tile(bboxes):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        center_x = x1 + (x2 - x1)/2
        center_y = y1 + (y2 - y1)/2

        center_x *= SCREENSHOT_WIDTH
        center_y *= SCREENSHOT_HEIGHT

        center_x = np.clip(center_x, a_min=TILE_INIT_X, a_max=SCREENSHOT_WIDTH - TILE_INIT_X)
        center_y = np.clip(center_x, a_min=TILE_INIT_Y, a_max=SCREENSHOT_HEIGHT - CARD_Y - TILE_INIT_Y)

        tile_x = np.round((center_x - TILE_INIT_X) / TILE_WIDTH)
        tile_y = np.round((center_y - TILE_INIT_Y) / TILE_HEIGHT)

        return tile_x, tile_y
    
    @staticmethod
    def tile_to_xy(tile_x, tile_y):
        x = (tile_x + 0.5) * TILE_WIDTH + TILE_INIT_X
        y = (tile_y + 0.5) * TILE_HEIGHT + TILE_INIT_Y

        return x, y

    def run(self, img, conf_thres=0.725, iou_thres=0.5):
        """
        Parameters
        ----------
        img : PIL.Image
            A PIL image with channels as last dimension. Pixel values should range from 0 to 255.
        """
        img_transform = self.preprocess(img)
        pred = self.sess.run([self.output_name], {self.input_name: img_transform})[0]

        pred = self.nms(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0]  # shape (M, 6)

        # transform relative coords in (UNIT_H, UNIT_W) to (height, width)
        pred[:, [0, 2]] *= SCREENSHOT_WIDTH / UNIT_W
        pred[:, [1, 3]] *= SCREENSHOT_HEIGHT / UNIT_H

        sides = self.calculate_side(img, pred)
        labels = pred[:, 5]

        return labels, pred[:, :4], sides


class SideDetector(OnnxDetector):

    def __init__(self, model_path):
        super().__init__(model_path)

    @staticmethod
    def preprocess(img):
        img = img.resize((SIDE_H, SIDE_W), Image.BICUBIC)
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        return img / 255

    def run(self, img):
        """
        Returns 0 or 1, 0 if unit present on image is in blue team, 1 otherwise.

        Parameters
        ----------
        img : PIL.Image
            A PIL image with channels as last dimension. Pixel values should range from 0 to 255.
        """
        img = self.preprocess(img)

        pred = self.sess.run([self.output_name], {self.input_name: img})[0]  # shape (1, 2)
        team = np.argmax(pred)

        return team
        