"""
Custom Random Forest Classifier Implementation
"""

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from .mytree import MyDecisionTreeClassifier


class MyRandomForestClassifier:
    """
    n_trees : int, default=20
        Number of trees to generate (N)
    n_selected_trees : int, default=7
        Number of best trees to select (M)
    n_features : int, default=2
        Number of random features to consider at each node (F)
    max_depth : int, default=10
        Maximum depth of each tree
    min_samples_split : int, default=2
        Minimum number of samples required to split a node
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_trees=20, n_selected_trees=7, n_features=2, 
                 max_depth=10, min_samples_split=2, random_state=None):
        self.n_trees = n_trees
        self.n_selected_trees = n_selected_trees
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.forest = []
        self.feature_indices_list = []
        self.test_set_indices = None
        self.remainder_set_indices = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _bootstrap_sample(self, X, y):
        """Generate a bootstrap sample from the data"""
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices], indices
    
    def _split_train_val(self, X, y, val_ratio=0.2):
        """Split bootstrap sample into training and validation sets"""
        n_samples = len(X)
        n_val = int(n_samples * val_ratio)
        indices = np.random.permutation(n_samples)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        
        return X_train, y_train, X_val, y_val
    
    def _get_random_features(self, n_total_features):
        """Get random feature indices for a node"""
        n_features_to_select = min(self.n_features, n_total_features)
        return np.random.choice(n_total_features, size=n_features_to_select, replace=False).tolist()
    
    def _calculate_accuracy(self, y_true, y_pred):
        """Calculate accuracy"""
        return np.mean(y_true == y_pred)
    
    def fit(self, X, y):
        """
        Fit the random forest classifier
        
        Pre-processing: Split data into 1/3 test and 2/3 remainder
        Generate N trees using bootstrapping, select M best trees
        """
        X = np.array(X)
        y = np.array(y)
        
        # Pre-processing: Split into test (1/3) and remainder (2/3)
        n_total = len(X)
        indices = np.arange(n_total)
        
        # Use StratifiedShuffleSplit to get indices
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1/3, random_state=self.random_state)
        remainder_indices, test_indices = next(sss.split(X, y))
        
        # Store indices for reference
        self.test_set_indices = test_indices
        self.remainder_set_indices = remainder_indices
        
        # Get actual data splits
        X_remainder = X[remainder_indices]
        X_test = X[test_indices]
        y_remainder = y[remainder_indices]
        y_test = y[test_indices]
        
        # Generate N trees
        trees_with_scores = []
        
        for i in range(self.n_trees):
            # Bootstrap sample from remainder set
            X_boot, y_boot, _ = self._bootstrap_sample(X_remainder, y_remainder)
            
            # Split into train and validation
            X_train, y_train, X_val, y_val = self._split_train_val(X_boot, y_boot)
            
            # Build tree with random feature selection at each node
            n_total_features = X_train.shape[1]
            feature_indices = self._get_random_features(n_total_features)
            
            # Build decision tree
            tree = MyDecisionTreeClassifier()
            tree.fit(X_train, y_train, feature_indices=feature_indices, 
                    max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            
            # Evaluate on validation set
            y_val_pred = tree.predict(X_val)
            accuracy = self._calculate_accuracy(y_val, y_val_pred)
            
            trees_with_scores.append((tree, accuracy, feature_indices))
        
        # Select M most accurate trees
        trees_with_scores.sort(key=lambda x: x[1], reverse=True)
        selected_trees = trees_with_scores[:self.n_selected_trees]
        
        self.forest = [tree for tree, _, _ in selected_trees]
        self.feature_indices_list = [feat_idx for _, _, feat_idx in selected_trees]
        
        return self
    
    def predict(self, X):
        """
        Predict class labels using majority voting
        """
        X = np.array(X)
        n_samples = X.shape[0]
        predictions = []
        
        for i in range(n_samples):
            # Get predictions from all trees in forest
            tree_predictions = []
            for tree in self.forest:
                pred = tree._predict_single(X[i], tree.tree)
                tree_predictions.append(pred)
            
            # Majority voting
            majority_class = Counter(tree_predictions).most_common(1)[0][0]
            predictions.append(majority_class)
        
        return np.array(predictions)
    
    def get_forest_size(self):
        """Get the number of trees in the forest"""
        return len(self.forest)

