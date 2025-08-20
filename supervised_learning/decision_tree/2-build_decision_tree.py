#!/usr/bin/env python3

import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_root = is_root
        self.is_leaf = False

    def left_child_add_prefix(self, text):
        lines = text.split('\n')
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |      " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        lines = text.split('\n')
        new_text = "    +---> " + lines + "\n"
        for x in lines[1:]:
            new_text += "           " + x + "\n"
        return new_text

    def __str__(self):
        if self.is_root:
            node_str = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            node_str = f"node [feature={self.feature}, threshold={self.threshold}]"
        result = node_str
        if self.left_child is not None:
            left_str = str(self.left_child)
            result += "\n" + self.left_child_add_prefix(left_str).rstrip()
        if self.right_child is not None:
            right_str = str(self.right_child)
            result += "\n" + self.right_child_add_prefix(right_str).rstrip()
        return result


class Leaf(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.is_leaf = True

    def __str__(self):
        return f"leaf [value={self.value}]"
