#!/usr/bin/env python3
"""
Module for Yolo v3 object detection
"""
import tensorflow.keras as K
import numpy as np


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

        Args:
            outputs: list of numpy.ndarrays containing predictions
                     Each output shape: (grid_height, grid_width,
                                        anchor_boxes, 4 + 1 + classes)
            image_size: numpy.ndarray containing original image size
                       [image_height, image_width]

        Returns:
            tuple of (boxes, box_confidences, box_class_probs):
            - boxes: list of numpy.ndarrays with processed boundary boxes
            - box_confidences: list of numpy.ndarrays with box confidences
            - box_class_probs: list of numpy.ndarrays with class probabilities
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

            # Apply sigmoid to box_confidence and box_class_prob
            box_confidence = 1 / (1 + np.exp(-box_confidence))
            box_class_prob = 1 / (1 + np.exp(-box_class_prob))

            # Create grid for center coordinates
            c_x = np.arange(grid_width).reshape(1, grid_width, 1)
            c_x = np.tile(c_x, [grid_height, 1, anchor_boxes])

            c_y = np.arange(grid_height).reshape(grid_height, 1, 1)
            c_y = np.tile(c_y, [1, grid_width, anchor_boxes])

            # Calculate box center coordinates (b_x, b_y)
            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_width
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_height

            # Get anchor dimensions for this output
            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]

            # Calculate box dimensions (b_w, b_h)
            # Input dimensions (from model.input.shape)
            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]

            b_w = (anchor_width * np.exp(t_w)) / input_width
            b_h = (anchor_height * np.exp(t_h)) / input_height

            # Convert to corner coordinates (x1, y1, x2, y2)
            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_x + b_w / 2) * image_width
            y2 = (b_y + b_h / 2) * image_height

            # Stack coordinates
            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return (boxes, box_confidences, box_class_probs)
