import cv2
import numpy as np


def draw_predicted_class(image, class_name, color, score, x, y, x_plus_w, y_plus_h):
    class_name_label = str(class_name)
    confidence_label = str(int(score * 100)) + '%'
    cv2.rectangle(image, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(image, class_name_label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(image, confidence_label, (x - 10, y - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_predicted_classes(image, indices, boxes, class_ids, classes, color, confidences):
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        class_id = class_ids[i]
        draw_predicted_class(image=image,
                             class_name=classes[class_id],
                             color=color[class_id],
                             score=confidences[i],
                             x=round(x),
                             y=round(y),
                             x_plus_w=round(x + w),
                             y_plus_h=round(y + h))


class PredictionDrawer:
    def __init__(self, prediction_result, classes):
        self.prediction_result = prediction_result
        self.classes = classes
        self.color = np.random.uniform(0, 255, size=(len(classes), 3))
        self.score_threshold = 0.5
        self.nms_threshold = 0.4

    def draw(self, image, image_width, image_height):
        class_ids = []
        scores = []
        bounding_boxes = []

        for out in self.prediction_result:
            for detection in out:
                confidences = detection[5:]
                class_id = np.argmax(confidences)
                score = confidences[class_id]

                if score > 0.5:
                    print('Found', self.classes[class_id], ' with confidence of', int(score * 100))
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)
                    w = int(detection[2] * image_width)
                    h = int(detection[3] * image_height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    scores.append(float(score))
                    bounding_boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(bboxes=bounding_boxes,
                                   scores=scores,
                                   score_threshold=self.score_threshold,
                                   nms_threshold=self.nms_threshold)

        for idx in indices:
            idx = idx[0]
            box = bounding_boxes[idx]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            class_id = class_ids[idx]
            draw_predicted_class(image=image,
                                 class_name=self.classes[class_id],
                                 color=self.color[class_id],
                                 score=scores[idx],
                                 x=round(x),
                                 y=round(y),
                                 x_plus_w=round(x + w),
                                 y_plus_h=round(y + h))


class Network:
    def __init__(self, weights_path, config_path):
        self.network = cv2.dnn.readNet(weights_path, config_path)

    def get_output_layers(self):
        layer_names = self.network.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]
        return output_layers

    def predict(self, image, scale=0.00392):
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        self.network.setInput(blob)
        return self.network.forward(self.get_output_layers())
