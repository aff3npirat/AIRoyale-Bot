from state.onnx_detector import OnnxDetector



class UnitDetector(OnnxDetector):
    
    def __init__(self, model_path):
        super().__init__(model_path)

    def _preprocess(self, img):
        return img / 255

    def run(self, img, conf_thres=0.725, iou_thres=0.5):
        img = self._preprocess(img)

        pred = self.sess.run([self.output_name], {self.input_name: img})[0]

        bboxes = self.nms(pred, conf_thres=conf_thres, iou_thres=iou_thres)  # shape (M, 6)

        return bboxes


class SideDetector(OnnxDetector):

    def __init__(self, model_path):
        super().__init__(model_path)

    def _preprocess(self, img):
        return img / 255

    def run(self, img):
        """
        Returns 0 or 1, 1 if unit present on image is in blue team, 0 otherwise.
        img: array with shape (M, SIDE_W, SIDE_H)
        """
        pred = self.sess.run([self.output_name], {self.input_name: img})

        return pred
        