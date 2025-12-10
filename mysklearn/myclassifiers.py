from math import sqrt
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
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
    
class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """

        self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train,y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_test_raw = self.regressor.predict(X_test)
        y_predicted = []
        for y_val in y_test_raw:
            y_predicted.append(self.discretizer(y_val))
        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """

        dimensions = len(X_test[0])

        distances = []
        neighbor_indices = []

        for point_index in range(len(X_test)):
            dist_ind = [] ##Distance + Index
            for other_point_index in range(len(self.X_train)):
                distance_between = 0
                for d_index in range(dimensions):
                    distance_between += (X_test[point_index][d_index] - self.X_train[other_point_index][d_index]) ** 2

                distance_between = round(sqrt(distance_between),3)
                dist_ind.append([distance_between, other_point_index])
            dist_ind.sort(key=lambda dist_ind: dist_ind[0])
            dist_ind = dist_ind[0:self.n_neighbors]

            closest_dist = []
            closest_ind = []
            for item in dist_ind:
                closest_dist.append(item[0])
                closest_ind.append(item[1])

            distances.append(closest_dist)
            neighbor_indices.append(closest_ind)

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        _, neighbor_indices = self.kneighbors(X_test)

        y_predicted = []

        for group_indices in neighbor_indices:
            class_counter = {}
            for index in group_indices:
                if self.y_train[index] in class_counter.keys():
                    class_counter[self.y_train[index]] += 1
                else:
                    class_counter[self.y_train[index]] = 0
            max_keys = [key for key, value in class_counter.items() if value == max(class_counter.values())] ##Did this in case of ties
            y_predicted.append(max_keys[0])
        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        label_counter = {}
        for label in y_train:
            if label in label_counter.keys():
                label_counter[label] += 1
            else:
                label_counter[label] = 0
        max_keys = [key for key, value in label_counter.items() if value == max(label_counter.values())] ##Did this in case of ties
        most_frequent_class = max_keys[0]
        self.most_common_label = most_frequent_class

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []
        for _ in range(len(X_test)):
            y_predicted.append(self.most_common_label)

        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        conditionals(YOU CHOOSE THE MOST APPROPRIATE TYPE): The conditional probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.conditionals = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        self.priors = {}
        self.conditionals = {}

        unique_labels = list(set(y_train))
        n_samples = len(y_train)
        n_features = len(X_train[0]) ##Number of attributes to check probability for

        
        for label in unique_labels:
            self.priors[label] = round(y_train.count(label) / n_samples,2) ##Calculates Priors
        
        
        for label in unique_labels:
            self.conditionals[label] = {}

            label_indices = [i for i in range(len(y_train)) if y_train[i] == label] ##Gives us all indices where y=label
            label_count = len(label_indices)

            for attribute_index in range(n_features):
                self.conditionals[label][attribute_index] = {}
                
                attr_values = [X_train[i][attribute_index] for i in label_indices] ##List of all values for index, for a given label
                
                for value in list(set(attr_values)):
                    count = attr_values.count(value)
                    self.conditionals[label][attribute_index][value] = round(count / label_count,2)

        pass

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        n_features = len(X_test[0])

        for row in X_test:
            posteriors = {}
            
            for label in self.priors:
                posterior = self.priors[label]
                
                # Multiply by conditionals for each attribute
                for attribute_index in range(n_features):
                    attr_value = row[attribute_index]
                    
                    if attr_value in self.conditionals[label][attribute_index]:
                        posterior *= self.conditionals[label][attribute_index][attr_value]
                    else:
                        posterior = 0
                        break
                
                posteriors[label] = posterior
            
            predicted_label = max(posteriors, key=posteriors.get)
            y_predicted.append(predicted_label)
        
        return y_predicted
