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

        Args:
            model_path: path to where a Darknet Keras model is stored
            classes_path: path to where the list of class names is found
            class_t: float representing the box score threshold for filtering
            nms_t: float representing the IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                     containing all of the anchor boxes
        """
        # Load the Darknet Keras model with compile=False
        self.model = K.models.load_model(model_path, compile=False)

        # Load class names from file
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Set thresholds
        self.class_t = class_t
        self.nms_t = nms_t

        # Set anchor boxes
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet outputs to get boundary boxes
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract box parameters
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]
            box_confidence = output[:, :, :, 4:5]
            box_class_prob = output[:, :, :, 5:]

            # Apply sigmoid
            box_confidence = 1 / (1 + np.exp(-box_confidence))
            box_class_prob = 1 / (1 + np.exp(-box_class_prob))

            # Create grid
            c_x = np.arange(grid_width).reshape(1, grid_width, 1)
            c_x = np.tile(c_x, [grid_height, 1, anchor_boxes])

            c_y = np.arange(grid_height).reshape(grid_height, 1, 1)
            c_y = np.tile(c_y, [1, grid_width, anchor_boxes])

            # Calculate centers
            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_width
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_height

            # Get anchor dimensions
            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]

            # Calculate dimensions
            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]

            b_w = (anchor_width * np.exp(t_w)) / input_width
            b_h = (anchor_height * np.exp(t_h)) / input_height

            # Convert to corners
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

        Args:
            filtered_boxes: numpy.ndarray of shape (?, 4)
            box_classes: numpy.ndarray of shape (?,)
            box_scores: numpy.ndarray of shape (?)

        Returns:
            tuple of (box_predictions, predicted_box_classes,
                     predicted_box_scores)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Get unique classes
        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            # Get indices for current class
            class_mask = box_classes == cls
            class_boxes = filtered_boxes[class_mask]
            class_scores = box_scores[class_mask]

            # Sort by scores (descending)
            sorted_indices = np.argsort(class_scores)[::-1]

            keep_indices = []

            while len(sorted_indices) > 0:
                # Keep the box with highest score
                current_idx = sorted_indices[0]
                keep_indices.append(current_idx)

                if len(sorted_indices) == 1:
                    break

                # Calculate IoU with remaining boxes
                current_box = class_boxes[current_idx]
                remaining_boxes = class_boxes[sorted_indices[1:]]

                iou = self._calculate_iou(current_box, remaining_boxes)

                # Keep only boxes with IoU less than threshold
                keep_mask = iou < self.nms_t
                sorted_indices = sorted_indices[1:][keep_mask]

            # Append kept boxes for this class
            kept_boxes = class_boxes[keep_indices]
            kept_scores = class_scores[keep_indices]
            kept_classes = np.full(len(keep_indices), cls)

            box_predictions.append(kept_boxes)
            predicted_box_classes.append(kept_classes)
            predicted_box_scores.append(kept_scores)

        # Concatenate results from all classes
        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        Load images from a folder

        Args:
            folder_path: string representing the path to the folder
                        holding all the images to load

        Returns:
            tuple of (images, image_paths):
            - images: list of images as numpy.ndarrays
            - image_paths: list of paths to the individual images
        """
        images = []
        image_paths = []

        # Get all files in the folder
        for filename in os.listdir(folder_path):
            # Construct full file path
            file_path = os.path.join(folder_path, filename)

            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                # Load image using cv2
                image = cv2.imread(file_path)

                # If image was loaded successfully, add it to lists
                if image is not None:
                    images.append(image)
                    image_paths.append(file_path)

        return (images, image_paths)

    def _calculate_iou(self, box, boxes):
        """
        Calculate Intersection over Union between one box and multiple boxes

        Args:
            box: numpy.ndarray of shape (4,) - single box [x1, y1, x2, y2]
            boxes: numpy.ndarray of shape (n, 4) - multiple boxes

        Returns:
            numpy.ndarray of shape (n,) containing IoU values
        """
        # Calculate intersection coordinates
        x1_inter = np.maximum(box[0], boxes[:, 0])
        y1_inter = np.maximum(box[1], boxes[:, 1])
        x2_inter = np.minimum(box[2], boxes[:, 2])
        y2_inter = np.minimum(box[3], boxes[:, 3])

        # Calculate intersection area
        inter_width = np.maximum(0, x2_inter - x1_inter)
        inter_height = np.maximum(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height

        # Calculate union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area

        return iou