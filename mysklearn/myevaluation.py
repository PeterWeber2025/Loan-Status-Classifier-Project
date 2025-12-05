import numpy as np # use numpy's random number generation
import math
from mysklearn import myutils


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    rng = np.random.default_rng(seed=random_state) #If random_state = None, default behavior is truly random

    if isinstance(test_size, float):
        test_sample_count = math.ceil(test_size * len(X))
    else:
        test_sample_count = test_size

    
    if shuffle:
        indices = np.arange(len(X))
        rng.shuffle(indices)

        test_indices = indices[:test_sample_count]
        train_indices = indices[test_sample_count:]

        x_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        x_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

    else:
        x_test = X[-test_sample_count:]
        y_test = y[-test_sample_count:]
        x_train = X[:-test_sample_count]
        y_train = y[:-test_sample_count]

    return x_train, x_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    rng = np.random.default_rng(seed=random_state) #If random_state = None, default behavior is truly random
    fold_length = []

    if len(X) % n_splits != 0:
        remainder = len(X) % n_splits
        for fold in range(n_splits):
            if remainder != 0:
                fold_length.append(len(X) // n_splits + 1)
                remainder -= 1
            else:
                fold_length.append(len(X) // n_splits)
    else:
        for fold in range(n_splits):
            fold_length.append(len(X) // n_splits)
    
    
    indices = np.arange(len(X))
    
    if shuffle:    
        rng.shuffle(indices)
    
    all_folds = []

    x_index = 0
    for length in fold_length:
        fold_values = []
        for index in range(length):
            fold_values.append(indices[x_index])
            x_index += 1
        all_folds.append(fold_values)
    
    fold_tuples = [] #List of tuples where [([x_test0],[x_train0]),[x_test1],[x_train1])]

    for i in range(n_splits):
        test_set = all_folds[i]
        train_set = []

        for fold in all_folds:
            if fold != test_set:
               for content in fold:
                   train_set.append(content) 
        fold_tupl = (train_set,test_set)
        fold_tuples.append(fold_tupl)

    return fold_tuples

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    return [] # TODO: (BONUS) fix this

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    rng = np.random.default_rng(seed=random_state) ##If random_state = None, default behavior is truly random

    if n_samples is None:
        n_samples = len(X)

    ##https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html
    sample_indices =  rng.choice(len(X), n_samples, replace=True)

    set_sample_indices = set(sample_indices) ##I do set() because the function is kind of slow, want it slightly faster
    out_of_bag_indices = [index for index in range(len(X)) if index not in set_sample_indices]

    X_sample = [X[index] for index in sample_indices]
    X_out_of_bag = [X[index] for index in out_of_bag_indices]

    if y is None:
        y_sample = None
        y_out_of_bag = None
    else:
        y_sample = [y[index] for index in sample_indices]
        y_out_of_bag = [y[index] for index in out_of_bag_indices]
        
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """ 
    
    matrix_row = [0 for _ in labels] 
    matrix = [matrix_row[:] for _ in labels] 

    #For every index, the value of y_true determines the row, and the value of y_pred determines the row
    #Then just add 1? 
    #
    for index in range(len(y_true)):
        row_index = labels.index(y_true[index])
        column_index = labels.index(y_pred[index])

        matrix[row_index][column_index] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    
    correct_predictions = 0
   
    for index in range(len(y_true)):
        if y_true[index] == y_pred[index]:
            correct_predictions += 1

    #print(correct_predictions)
    if normalize:
        correct_predictions = float(correct_predictions / len(y_true))

    return correct_predictions

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    #print(pos_label)
    # print("y_true:", y_true[0:10])
    # print("y_pred:", y_pred[0:10])
    # print("pos_label:", pos_label)

    true_positive = 0
    false_positive = 0

    for index in range(len(y_true)):
        if y_true[index] == pos_label:
            if y_pred[index] == pos_label:
                true_positive += 1
        else:
            if y_pred[index] == pos_label:
                false_positive += 1

    if true_positive == 0 and false_positive == 0:
        precision = 0.0
        print(true_positive, false_positive)
    else:
        precision = true_positive / (true_positive + false_positive)

    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    true_positive = 0
    false_negative = 0

    for index in range(len(y_true)):
        if y_true[index] == pos_label:
            if y_pred[index] == pos_label:
                true_positive += 1
            else:
                false_negative += 1
    
    if true_positive == 0 and false_negative == 0:
        recall = 0.0
        print(true_positive, false_negative)
    else:
        recall = true_positive / (true_positive + false_negative) 

    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)

    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)

    return f1_score
