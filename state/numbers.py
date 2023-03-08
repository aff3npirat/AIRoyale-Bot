import numpy as np
from PIL import Image

from constants import ELIXIR_BOUNDING_BOX, NUMBER_WIDTH, NUMBER_HEIGHT, TOWER_HP_BOXES, KING_HP, PRINCESS_HP
from state.onnx_detector import OnnxDetector



class NumberDetector(OnnxDetector):

    @staticmethod
    def calculate_elixir(image):
            crop = image.crop(ELIXIR_BOUNDING_BOX)
            std = np.array(crop).std(axis=(0, 2))
            rolling_std = np.convolve(std, np.ones(10) / 10, mode='valid')
            change_points = np.nonzero(rolling_std < 50)[0]
            if len(change_points) == 0:
                elixir = 10
            else:
                elixir = (change_points[0] + 10) // 25
            return elixir
    
    @staticmethod
    def relative_tower_hp(pred, king_level):
        for team in ["ally", "enemy"]:
            level = king_level[team]
            pred[f"{team}_king_hp"]["number"] /= KING_HP[level-1]

            max_princess_hp = PRINCESS_HP[level-1]
            for side in ["right", "left"]:
                pred[f"{side}_{team}_princess_hp"]["number"] /= max_princess_hp

    @staticmethod
    def _calculate_confidence_and_number(pred):
        pred = pred.tolist()[:4]  # only take 4 best predictions
        pred.sort(key=lambda x: x[0])

        confidence = [p[4] for p in pred]
        number = ''.join([str(int(p[5])) for p in pred])

        confidence = confidence if confidence else [-1]
        number = int(number) if number != '' else -1

        return confidence, number

    def post_process(self, pred, bboxes):
        clean_pred = {}
        for p, (name, bounding_box) in zip(pred, bboxes):
            confidence, number = self._calculate_confidence_and_number(p)
            clean_pred[name] = {'bounding_box': bounding_box,
                                'confidence': confidence,
                                'number': number}

        return clean_pred
    
    @staticmethod
    def preprocess(image):
        # Resize the image
        image = image.resize((NUMBER_WIDTH, NUMBER_HEIGHT), Image.Resampling.BICUBIC)

        # Convert the image to grayscale
        image = np.array(image, dtype=np.float32)
        gray = np.dot(image[:, :, :3], [0.2125, 0.7154, 0.0721])
        for i in range(3):
            image[:, :, i] = gray

        # Add padding
        padded_image = 114 * np.ones((NUMBER_WIDTH, NUMBER_WIDTH, 3), dtype=np.float32)
        top = (NUMBER_WIDTH - NUMBER_HEIGHT) // 2
        padded_image[top: top + NUMBER_HEIGHT, :, :] = image

        padded_image = padded_image / 255
        padded_image = np.expand_dims(padded_image.transpose(2, 0, 1), axis=0)
        return padded_image
    
    def run(self, image, conf_thres=0.25, iou_thres=0.45):
        # Preprocessing
        crops = np.empty((len(TOWER_HP_BOXES), 3, NUMBER_WIDTH, NUMBER_WIDTH), dtype=np.float32)
        for i, (_, bounding_box) in enumerate(TOWER_HP_BOXES):
            crop = image.crop(bounding_box)
            crops[i] = self.preprocess(crop)

        # Inference
        pred = self.sess.run([self.output_name], {self.input_name: crops})[0]

        # Forced post-processing
        pred = self.nms(pred, conf_thres=conf_thres, iou_thres=iou_thres)

        # Custom post-processing
        pred = self.post_process(pred, TOWER_HP_BOXES)

        # Elixir
        pred['elixir'] = {'bounding_box': ELIXIR_BOUNDING_BOX,
                          'confidence': 1.0,
                          'number': self.calculate_elixir(image)}
        return pred
    
