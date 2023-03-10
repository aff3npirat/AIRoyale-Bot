import numpy as np
from PIL import Image

from state.onnx_detector import OnnxDetector
from constants import (
    SIDE_H,
    SIDE_W,
    UNIT_H,
    UNIT_W,
    UNIT_Y_START,
    UNIT_Y_END,
    UNIT_NAMES,
    CARD_TO_UNITS,
    TILE_INIT_X,
    TILE_INIT_Y,
    TILE_END_Y,
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
        img = img.crop((0, UNIT_Y_START, img.width, UNIT_Y_END))
        img = img.resize((UNIT_H, UNIT_W), Image.Resampling.BICUBIC)
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
    def xy_to_tile(x, y):
        """
        Expects bboxes in absolute coordinates in screenshot.
        """
        x = np.clip(x, a_min=TILE_INIT_X, a_max=SCREENSHOT_WIDTH - TILE_INIT_X)
        y = np.clip(y, a_min=TILE_INIT_Y, a_max=TILE_END_Y)

        tile_x = np.trunc((x - TILE_INIT_X) / TILE_WIDTH)
        tile_y = np.trunc((y - TILE_INIT_Y) / TILE_HEIGHT)

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
        pred = self.sess.run([self.output_name], {self.input_name: img_transform})[0]  # returns absolute coords

        pred = self.nms(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0]  # shape (num_boxes, 6)

        # get absolute coords in unprocessed image
        pred[:, [0, 2]] *= SCREENSHOT_WIDTH / UNIT_W
        pred[:, [1, 3]] *= (UNIT_Y_END - UNIT_Y_START) / UNIT_H
        pred[:, [1, 3]] += UNIT_Y_START

        sides = self.calculate_side(img, pred)
        labels = pred[:, 5]

        return labels, pred[:, :4], sides


class SideDetector(OnnxDetector):

    def __init__(self, model_path):
        super().__init__(model_path)

    @staticmethod
    def preprocess(img):
        img = img.resize((SIDE_H, SIDE_W), Image.Resampling.BICUBIC)
        img = np.array(img, dtype=np.float32).transpose(2, 0, 1)
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

        team = self.sess.run([self.output_name], {self.input_name: img})[0][:, 0]  # shape (batch_size,)

        return team
        