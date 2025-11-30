"""
Custom Decision Tree Classifier Implementation
Used as the base for Random Forest
"""

import numpy as np
from collections import Counter


class MyDecisionTreeClassifier:
    """Decision Tree Classifier using entropy for splitting"""
    
    def __init__(self):
        self.tree = None
        
    def _entropy(self, y):
        """Calculate entropy of a set of labels"""
        if len(y) == 0:
            return 0
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
        return entropy
    
    def _information_gain(self, y_parent, y_left, y_right):
        """Calculate information gain from a split"""
        parent_entropy = self._entropy(y_parent)
        n = len(y_parent)
        if n == 0:
            return 0
        left_weight = len(y_left) / n
        right_weight = len(y_right) / n
        weighted_entropy = left_weight * self._entropy(y_left) + right_weight * self._entropy(y_right)
        return parent_entropy - weighted_entropy
    
    def _find_best_split(self, X, y, feature_indices):
        """Find the best split among the given feature indices"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            # Get unique values for this feature
            values = sorted(set(X[:, feature_idx]))
            
            # Try thresholds between consecutive values
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Calculate information gain
                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, feature_indices, max_depth=10, min_samples_split=2, depth=0):
        """Recursively build the decision tree"""
        # Base cases
        if len(y) == 0:
            return None
        
        # If all labels are the same, return a leaf
        if len(set(y)) == 1:
            return {'leaf': True, 'class': y[0]}
        
        # If max depth reached or too few samples, return majority class
        if depth >= max_depth or len(y) < min_samples_split:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        # If no features available, return majority class
        if len(feature_indices) == 0:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, feature_indices)
        
        # If no good split found, return majority class
        if best_feature is None or best_gain <= 0:
            return {'leaf': True, 'class': Counter(y).most_common(1)[0][0]}
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Build subtrees
        left_tree = self._build_tree(
            X[left_mask], y[left_mask], feature_indices,
            max_depth, min_samples_split, depth + 1
        )
        right_tree = self._build_tree(
            X[right_mask], y[right_mask], feature_indices,
            max_depth, min_samples_split, depth + 1
        )
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y, feature_indices=None, max_depth=10, min_samples_split=2):
        X = np.array(X)
        y = np.array(y)
        
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
        
        self.tree = self._build_tree(X, y, feature_indices, max_depth, min_samples_split)
        return self
    
    def _predict_single(self, x, tree):
        """Predict a single instance"""
        if tree is None:
            return None
        
        if tree['leaf']:
            return tree['class']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])
    
    def predict(self, X):
        """Predict class labels for instances in X"""
        X = np.array(X)
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.tree))
        return np.array(predictions)

