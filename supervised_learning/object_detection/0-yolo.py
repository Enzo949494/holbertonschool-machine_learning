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
