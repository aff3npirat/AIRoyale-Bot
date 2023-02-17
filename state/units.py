from state.onnx_detector import OnnxDetector



class UnitDetector(OnnxDetector):
    
    def __init__(self, model_path):
        super().__init__(model_path)

    def _preprocess(self, img):
        return img / 255

    def run(self, img, conf_thres=0.725, iou_thres=0.5):
        img = self._preprocess(img)

        pred = self.sess.run([self.output_name], {self.input_name: img})[0]

        bboxes = self.nms(pred, conf_thres=conf_thres, iou_thres=iou_thres)

        return bboxes


class SideDetector:
    pass