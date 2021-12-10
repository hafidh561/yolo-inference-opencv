import cv2
import random
import numpy as np


class YoloOpenCV:
    def __init__(
        self,
        weights="./models/coco-dataset/yolov4.weights",
        config="./models/coco-dataset/yolov4.cfg",
        obj_names="./models/coco-dataset/obj.names",
        size_model=608,
        device="cpu",
        conf_thresh=0.5,
    ):
        self.weights = str(weights)
        self.config = str(config)
        self.obj_names = str(obj_names)
        self.size_model = int(size_model)
        self.device = str(device)
        self.conf_thresh = float(conf_thresh)

        self.nms_threshold = 0 if self.conf_thresh - 0.1 < 0 else self.conf_thresh - 0.1
        with open(self.obj_names, "r") as f:
            self.class_names = f.read().splitlines()
            self.class_colors = [
                self.__generate_random_color() for _ in self.class_names
            ]
        self.net = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        if self.device == "cpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif self.device == "cuda":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            raise ValueError("Please input between 'cpu' and 'cuda'")

    def predict(self, img):
        image = img
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255, (self.size_model, self.size_model), (0, 0, 0), 1, crop=False
        )
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_names = [
            layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
        ]
        outputs = self.net.forward(output_names)
        final_img = self.__postprocess_result(outputs, image)
        return final_img

    def __postprocess_result(self, outputs, img):
        image = img
        hT, wT, _ = image.shape
        bbox = []
        class_ids = []
        confidences = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_thresh:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    class_ids.append(class_id)
                    confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(
            bbox, confidences, self.conf_thresh, self.nms_threshold
        )
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(
                image, (x, y), (x + w, y + h), self.class_colors[class_ids[i]], 2
            )
            cv2.putText(
                image,
                f"{self.class_names[class_ids[i]].upper()} {int(confidences[i]*100)}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                self.class_colors[class_ids[i]],
                2,
            )
        return image

    def __generate_random_color(self):
        return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
