import numpy as np
from PIL import Image

from state.onnx_detector import OnnxDetector
from constants import SIDE_H, SIDE_W, UNIT_H, UNIT_W



class UnitDetector(OnnxDetector):
    
    def __init__(self, model_path):
        super().__init__(model_path)

    @staticmethod
    def preprocess(img):
        img = img.resize((UNIT_H, UNIT_W), Image.BICUBIC)
        img = np.array(img, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img / 255

    def run(self, img, conf_thres=0.725, iou_thres=0.5):
        """
        Returns 0 or 1, 0 if unit present on image is in blue team, 1 otherwise.

        Parameters
        ----------
        img : PIL.Image
            A PIL image with channels as last dimension. Pixel values should range from 0 to 255.
        """
        img = self._preprocess(img)

        pred = self.sess.run([self.output_name], {self.input_name: img})[0]

        bboxes = self.nms(pred, conf_thres=conf_thres, iou_thres=iou_thres)  # shape (M, 6)

        return bboxes


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
        img = self._preprocess(img)

        pred = self.sess.run([self.output_name], {self.input_name: img})[0]  # shape (1, 2)
        team = np.argmax(pred)

        return team
        