#!/usr/bin/env python3
"""
Decision tree build module for machine learning.
This module provides classes to build and manage decision trees.
"""

import numpy as np


class Node:
    """
    A node in a decision tree.

    Represents an internal node that makes decisions based on
    feature values and thresholds.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialize a Node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calculate the maximum depth below this node.
        """
        max_depth = self.depth
        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes below this node.
        """
        count = 0 if only_leaves else 1

        if self.left_child is not None:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child is not None:
            count += self.right_child.count_nodes_below(only_leaves)

        return count

    def get_leaves_below(self):
        """
        Get all leaves below this node.
        """
        leaves = []

        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())

        return leaves

    def update_bounds_below(self):
        """
        Update bounds for this node and all nodes below.
        """
        if self.is_root:
            # Initialiser pour TOUTES les features
            n_features = self.explanatory.shape[1] if hasattr(self, 'explanatory') else 10
            self.upper = {i: np.inf for i in range(n_features)}
            self.lower = {i: -np.inf for i in range(n_features)}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()
                
                if child == self.left_child:
                    child.upper[self.feature] = self.threshold
                else:
                    child.lower[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """
        Update the indicator function for this node.
        """
        def is_large_enough(x):
            if not hasattr(self, 'lower') or not self.lower:
                return np.ones(x.shape[0], dtype=bool)

            conditions = []
            for key in self.lower.keys():
                if self.lower[key] == -np.inf:
                    conditions.append(np.ones(x.shape[0], dtype=bool))
                else:
                    conditions.append(
                        np.greater(x[:, key], self.lower[key]))

            return np.all(np.array(conditions), axis=0)

        def is_small_enough(x):
            if not hasattr(self, 'upper') or not self.upper:
                return np.ones(x.shape[0], dtype=bool)

            conditions = []
            for key in self.upper.keys():
                if self.upper[key] == np.inf:
                    conditions.append(np.ones(x.shape[0], dtype=bool))
                else:
                    conditions.append(
                        np.less_equal(x[:, key], self.upper[key]))

            return np.all(np.array(conditions), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Predict class for a single sample using this node.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

    def left_child_add_prefix(self, text):
        """
        Add prefix for left child in tree visualization.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Add prefix for right child in tree visualization.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """
        String representation of the node and its subtree.
        """
        if self.is_root:
            node_str = (f"root [feature={self.feature}, "
                        f"threshold={self.threshold}]")
        else:
            node_str = (f"-> node [feature={self.feature}, "
                        f"threshold={self.threshold}]")

        result = node_str

        if self.left_child is not None:
            left_str = str(self.left_child)
            result += "\n" + self.left_child_add_prefix(left_str).rstrip()

        if self.right_child is not None:
            right_str = str(self.right_child)
            result += "\n" + self.right_child_add_prefix(right_str).rstrip()

        return result


class Leaf(Node):
    """
    A leaf node in a decision tree.
    """

    def __init__(self, value, depth=0):
        """
        Initialize a Leaf.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of this leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count this leaf as one node.
        """
        return 1

    def get_leaves_below(self):
        """
        Return this leaf in a list.
        """
        return [self]

    def update_bounds_below(self):
        """
        Update bounds for leaf node (no operation needed).
        """
        pass

    def update_indicator(self):
        """
        Update the indicator function for this leaf.
        """
        def is_large_enough(x):
            if not hasattr(self, 'lower') or not self.lower:
                return np.ones(x.shape[0], dtype=bool)

            conditions = []
            for key in self.lower.keys():
                if self.lower[key] == -np.inf:
                    conditions.append(np.ones(x.shape[0], dtype=bool))
                else:
                    conditions.append(
                        np.greater(x[:, key], self.lower[key]))

            return np.all(np.array(conditions), axis=0)

        def is_small_enough(x):
            if not hasattr(self, 'upper') or not self.upper:
                return np.ones(x.shape[0], dtype=bool)

            conditions = []
            for key in self.upper.keys():
                if self.upper[key] == np.inf:
                    conditions.append(np.ones(x.shape[0], dtype=bool))
                else:
                    conditions.append(
                        np.less_equal(x[:, key], self.upper[key]))

            return np.all(np.array(conditions), axis=0)

        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Predict class for a single sample using this leaf.
        """
        return self.value

    def __str__(self):
        """
        String representation of the leaf.
        """
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """
    A decision tree classifier.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize a Decision Tree.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Get the depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves)

    def get_leaves(self):
        """
        Get all leaves in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Update bounds for all nodes in the tree.
        """
        # Passer l'info sur le nombre de features à la racine
        self.root.explanatory = self.explanatory
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Update the prediction function for the decision tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_function(A):
            predictions = np.zeros(A.shape[0])
            for leaf in leaves:
                mask = leaf.indicator(A)
                predictions[mask] = leaf.value
            return predictions.astype(int)

        self.predict = predict_function

    def pred(self, x):
        """
        Predict class for a single sample.
        """
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """
        Train the decision tree.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            train_accuracy = self.accuracy(self.explanatory, self.target)
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {train_accuracy}""")

    def np_extrema(self, arr):
        """
        Return min and max of array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Random split criterion.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """
        Fit a single node.
        """
        node.feature, node.threshold = self.split_criterion(node)

        # Split population based on threshold
        feature_values = self.explanatory[:, node.feature][node.sub_population]
        left_population = node.sub_population.copy()
        right_population = node.sub_population.copy()

        # CORRECTION: Left: feature > threshold, Right: feature <= threshold
        left_mask = feature_values > node.threshold
        right_mask = feature_values <= node.threshold

        # Update populations
        indices = np.where(node.sub_population)[0]
        left_population[indices[right_mask]] = False
        right_population[indices[left_mask]] = False

        # Check if left node should be a leaf
        left_targets = self.target[left_population]
        is_left_leaf = (
            np.sum(left_population) <= self.min_pop or
            node.depth + 1 >= self.max_depth or
            len(np.unique(left_targets)) == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Check if right node should be a leaf
        right_targets = self.target[right_population]
        is_right_leaf = (
            np.sum(right_population) <= self.min_pop or
            node.depth + 1 >= self.max_depth or
            len(np.unique(right_targets)) == 1
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Create a leaf child.
        """
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Create a node child.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Calculate accuracy.
        """
        predictions = self.predict(test_explanatory)
        return (np.sum(np.equal(predictions, test_target)) /
                test_target.size)

    def __str__(self):
        """
        String representation of the tree.
        """
        return self.root.__str__()
