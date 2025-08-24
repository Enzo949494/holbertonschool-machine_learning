#!/usr/bin/env python3
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree
import numpy as np

class Random_Forest():
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Predict classes using voting from all trees in the forest.
        
        Args:
            explanatory (numpy.ndarray): Input features to predict.
            
        Returns:
            numpy.ndarray: Predicted classes based on majority vote.
        """
        # Initialize an empty list to store predictions from individual trees
        all_predictions = []
        
        # Generate predictions for each tree in the forest
        for tree_predict in self.numpy_preds:
            predictions = tree_predict(explanatory)
            all_predictions.append(predictions)
        
        # Convert to numpy array for easier manipulation
        # Shape: (n_trees, n_samples)
        all_predictions = np.array(all_predictions)
        
        # Calculate the mode (most frequent) prediction for each example
        # For each sample, find the most frequent prediction across all trees
        final_predictions = []
        for i in range(explanatory.shape[0]):
            # Get predictions from all trees for sample i
            sample_predictions = all_predictions[:, i]
            # Find the most frequent prediction (mode)
            values, counts = np.unique(sample_predictions, return_counts=True)
            most_frequent = values[np.argmax(counts)]
            final_predictions.append(most_frequent)
        
        return np.array(final_predictions)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop, 
                            seed=self.seed+i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory, self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        return np.sum(np.equal(self.predict(test_explanatory), test_target))/test_target.size