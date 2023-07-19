import onnxruntime
import torch
import torchvision
import numpy as np



class OnnxDetector:
    """
    Base class for all detectors.

    Uses onnxruntime to perform inference.
    """

    def __init__(self, model_path):
        self.sess = onnxruntime.InferenceSession(model_path)

        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    @staticmethod
    def _xywh_to_xyxy(boxes):
        xyxy = boxes.copy()
        xyxy[..., 0] = boxes[..., 0] - boxes[..., 2]/2
        xyxy[..., 1] = boxes[..., 1] - boxes[..., 3]/2
        xyxy[..., 2] = boxes[..., 0] + boxes[..., 2]/2
        xyxy[..., 3] = boxes[..., 1] + boxes[..., 3]/2

        return xyxy

    @staticmethod
    def nms(prediction, conf_thres=0.25, iou_thres=0.45, max_wh=7800):
        """
        Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
        
        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        mi = 5 + nc  # mask start index
        output = [np.zeros((0, 6))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = OnnxDetector._xywh_to_xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)

            conf = x[:, 5:mi].max(1, keepdims=True)  # conf
            j = x[:, 5:mi].argmax(1, keepdims=True)  # class labels
            x = np.concatenate((box, conf, j.astype(float)), 1)[conf.reshape(-1) > conf_thres]


            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[x[:, 4].argsort()[::-1]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes

            # box coordinates are normed to intervall [0, 1], by offsetting with class labels (0, 1, 2,...)
            # no boxes overlap that do not belong to the same class
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_thres)  # NMS

            x_select = x[i]
            if len(x_select.shape) == 1:
                x_select = np.expand_dims(x_select, 0)

            output[xi] = x_select

        return output
