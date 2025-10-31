#!/usr/bin/env python3
"""
Module for Yolo v3 object detection
"""
import tensorflow.keras as K
import numpy as np
import cv2
import os


class Yolo:
    """
    Class that uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize Yolo class
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet outputs to get boundary boxes
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        if isinstance(self.model.input, list):
            input_width = self.model.input[0].shape[1]
            input_height = self.model.input[0].shape[2]
        else:
            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]
            box_confidence = output[:, :, :, 4:5]
            box_class_prob = output[:, :, :, 5:]

            box_confidence = 1 / (1 + np.exp(-box_confidence))
            box_class_prob = 1 / (1 + np.exp(-box_class_prob))

            c_x = np.arange(grid_width).reshape(1, grid_width, 1)
            c_x = np.tile(c_x, [grid_height, 1, anchor_boxes])

            c_y = np.arange(grid_height).reshape(grid_height, 1, 1)
            c_y = np.tile(c_y, [1, grid_width, anchor_boxes])

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_width
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_height

            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]

            b_w = (anchor_width * np.exp(t_w)) / input_width
            b_h = (anchor_height * np.exp(t_h)) / input_height

            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_x + b_w / 2) * image_width
            y2 = (b_y + b_h / 2) * image_height

            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes based on box score threshold
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box_score = box_confidences[i] * box_class_probs[i]
            box_class = np.argmax(box_score, axis=-1)
            box_class_score = np.max(box_score, axis=-1)
            filtering_mask = box_class_score >= self.class_t

            filtered_boxes.append(boxes[i][filtering_mask])
            box_classes.append(box_class[filtering_mask])
            box_scores.append(box_class_score[filtering_mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply Non-Maximum Suppression to filtered boxes
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            class_mask = box_classes == cls
            class_boxes = filtered_boxes[class_mask]
            class_scores = box_scores[class_mask]

            sorted_indices = np.argsort(class_scores)[::-1]

            keep_indices = []

            while len(sorted_indices) > 0:
                current_idx = sorted_indices[0]
                keep_indices.append(current_idx)

                if len(sorted_indices) == 1:
                    break

                current_box = class_boxes[current_idx]
                remaining_boxes = class_boxes[sorted_indices[1:]]

                iou = self._calculate_iou(current_box, remaining_boxes)

                keep_mask = iou < self.nms_t
                sorted_indices = sorted_indices[1:][keep_mask]

            kept_boxes = class_boxes[keep_indices]
            kept_scores = class_scores[keep_indices]
            kept_classes = np.full(len(keep_indices), cls)

            box_predictions.append(kept_boxes)
            predicted_box_classes.append(kept_classes)
            predicted_box_scores.append(kept_scores)

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        Load images from a folder
        """
        images = []
        image_paths = []

        for filename in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                image = cv2.imread(file_path)

                if image is not None:
                    images.append(image)
                    image_paths.append(file_path)

        return (images, image_paths)

    def preprocess_images(self, images):
        """
        Preprocess images for the Darknet model
        """
        if isinstance(self.model.input, list):
            input_w = self.model.input[0].shape[1]
            input_h = self.model.input[0].shape[2]
        else:
            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append([image.shape[0], image.shape[1]])

            resized = cv2.resize(image, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)

            rescaled = resized / 255.0

            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return (pimages, image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Display image with boundary boxes, class names, and box scores

        Args:
            image: numpy.ndarray containing an unprocessed image
            boxes: numpy.ndarray containing the boundary boxes for the image
            box_classes: numpy.ndarray containing the class indices for each
                        box
            box_scores: numpy.ndarray containing the box scores for each box
            file_name: the file path where the original image is stored
        """
        # Make a copy to avoid modifying the original image
        img_display = image.copy()

        # Iterate through all boxes
        for i in range(len(boxes)):
            # Get box coordinates
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw blue rectangle (thickness 2)
            cv2.rectangle(img_display, (x1, y1), (x2, y2),
                          (255, 0, 0), 2)

            # Get class name and score
            class_name = self.class_names[int(box_classes[i])]
            score = box_scores[i]

            # Format text: "ClassName Score"
            label = "{} {:.2f}".format(class_name, score)

            # Position text 5 pixels above the box
            text_x = x1
            text_y = y1 - 5

            # Draw red text
            cv2.putText(img_display, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)

        # Display the image
        cv2.imshow(file_name, img_display)

        # Wait for key press
        key = cv2.waitKey(0)

        # If 's' key is pressed, save the image
        if key == ord('s'):
            # Create detections directory if it doesn't exist
            if not os.path.exists('detections'):
                os.makedirs('detections')

            # Save image in detections directory
            save_path = os.path.join('detections', file_name)
            cv2.imwrite(save_path, img_display)

        # Close the window
        cv2.destroyAllWindows()

    def _calculate_iou(self, box, boxes):
        """
        Calculate Intersection over Union between one box and multiple boxes
        """
        x1_inter = np.maximum(box[0], boxes[:, 0])
        y1_inter = np.maximum(box[1], boxes[:, 1])
        x2_inter = np.minimum(box[2], boxes[:, 2])
        y2_inter = np.minimum(box[3], boxes[:, 3])

        inter_width = np.maximum(0, x2_inter - x1_inter)
        inter_height = np.maximum(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area

        iou = inter_area / union_area

        return iou
