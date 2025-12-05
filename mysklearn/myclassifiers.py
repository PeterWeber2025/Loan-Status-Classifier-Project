import math
from math import sqrt
from collections import defaultdict
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn import myutils


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.train = None
        self.tree = None
        self.header = None

    def calculate_weighted_partition_entropy(self, instances, attribute, header, attribute_domains):
        '''
        Returns the weighted average entropy of a set of partitions after splitting instances on an attribute
        '''
        total_rows = len(instances)
        att_index = header.index(attribute)
        
        att_domain = attribute_domains[attribute]
        
        weighted_entropy = 0
        partition_entropy = []

        for att_value in att_domain: ## For Every Unique Value In The Attribute
            att_value_rows = [row for row in instances if row[att_index] == att_value] ##Get All Rows Where Unique Value Occurs
            class_value_rows = [row[-1] for row in instances if row[att_index] == att_value]
            unique_class_values = list(set(class_value_rows))
            if len(att_value_rows) == 0: 
                continue

            entropy = 0

            for unique_class in unique_class_values: ## For Each Unique Class Label
                class_amount = 0
                for row in att_value_rows:
                    if row[-1] == unique_class: ##Add Up The Number Of Times It Occurs Under Unique Attribute Value
                        class_amount += 1

                probability = class_amount / len(att_value_rows)
                if probability > 0:
                    entropy -=  probability * math.log(probability, 2) ##entropy -= because the formula involves multiplying by '-'

            partition_entropy.append((entropy, len(att_value_rows))) ##Append each Partitions Entropy And size To List.
        
        ##With The Entropies For Each Partition Of Attribute Found, It Is Time For Calculating weighted Entropy

        for partition in partition_entropy: #0 is entropy, 1 is size of partition.
            weighted_entropy += partition[0] * (partition[1]/total_rows)

        return weighted_entropy    

    def select_attribute(self, instances, attributes, header, attribute_domains):
        '''
        Basically iterates through each attribute calculating the entropy of splitting on it.
        All Entropies are saved, the atrribute that leads to the lowest entropy after splitting is the returned attribute.
        
        :param instances: All the rows of data to consider splits on.
        :param attributes: Attributes We we can split on
        '''

        weighted_entropies = []

        for attribute in attributes:
            weighted_entropy = self.calculate_weighted_partition_entropy(instances,attribute, header, attribute_domains)
            weighted_entropies.append(weighted_entropy)

        minimum_entropy = min(weighted_entropies)
        best_attribute_index = weighted_entropies.index(minimum_entropy)

        best_attribute = attributes[best_attribute_index]
        #print(attributes)
        #print(weighted_entropies)
        return best_attribute

    def partition_instances(self, instances, attribute, header, attribute_domains):
        # this is group by attribute domain (not values of attribute in instances)
        # Returns a dictionary: {attribute_value: [instances]}
        att_index = header.index(attribute)
        att_domain = attribute_domains[attribute]
        partitions = {}
        for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)
        return partitions

    def tdidt(self, current_instances, available_attributes, header):  
        attribute_domains = myutils.find_attribute_domains(current_instances, available_attributes, header)
        split_attribute = self.select_attribute(current_instances, available_attributes, header, attribute_domains)
        available_attributes.remove(split_attribute) # can't split on this attribute again in this subtree

        tree = ["Attribute", split_attribute]

        partitions = self.partition_instances(current_instances, split_attribute, header, attribute_domains)
        #print("partitions:", list(partitions.keys()))

        # for each partition, repeat unless one of the following base cases occurs
        for att_value in sorted(partitions.keys()): # process in alphabetical order
            att_value_partition = partitions[att_value]
            value_subtree = ["Value", att_value]


            if len(att_value_partition) > 0 and myutils.all_same_class(att_value_partition): ##CASE 1: all class labels of the partition are the same, make leaf
                #print("CASE 1")
                #print(att_value_partition)
                value_subtree.append(["Leaf", att_value_partition[0][-1], len(att_value_partition), len(current_instances)])
                tree.append(value_subtree)

            #    CASE 2: no more attributes to select (clash)
            # => handle clash w/majority vote leaf node
            elif len(att_value_partition) > 0 and len(available_attributes) == 0:
                #print("CASE 2")
                class_frequency = {}
                for row in att_value_partition:
                    value = row[-1]
                    class_frequency[value] = class_frequency.get(value, 0) + 1
                chosen_class = max(class_frequency, key=class_frequency.get)
                value_subtree.append(["Leaf", chosen_class, len(att_value_partition), len(current_instances)])
                tree.append(value_subtree)
            
            elif len(att_value_partition) == 0: ## CASE 3: Empty partition backtrack and replace attribute node with majority vote leaf node
                #print("CASE 3")
                class_frequency = {}
                for row in att_value_partition:
                    value = row[-1]
                    class_frequency[value] = class_frequency.get(value, 0) + 1
                chosen_class = max(class_frequency, key=class_frequency.get)
                return ["Leaf", chosen_class, len(current_instances), len(current_instances)]
                ##I return here because it's the easiest way to remove prior attribute.
            
            else:
                subtree = self.tdidt(att_value_partition, available_attributes.copy(), header)
                value_subtree.append(subtree)
                tree.append(value_subtree)
        return tree

    def fit(self, X_train, y_train, header=None):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        if header is None:
            header = []    
            for x_attr_index in range(len(X_train[0])):
                header.append('att' + str(x_attr_index))
                self.header = header
        else:
            self.header=header

        self.train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        self.tree = self.tdidt(self.train, header.copy(), header.copy()) ##First Copy Turns Into Available Attributes, other stays header
        #print(self.tree)
        pass

    def predict_row(self, tree, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        data_type = tree[0]
        
        # Base case: if this is a leaf, just return its class label
        if data_type == "Leaf":
            label = tree[1]
            return label
        
        # Recursive case:if we are here, this is an Attribute node
        attribute_name = tree[1]
        attribute_index = self.header.index(attribute_name)
        instance_value = X_test[attribute_index]

        # Look for the matching value node
        for values in tree[2:]:
            value = values[1]
            subtree = values[2]
            
            if instance_value == value:
                return self.predict_row(subtree, X_test)

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            predictions.append(self.predict_row(self.tree, row))
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            attribute_names = self.header
        
        if self.tree[0] == "Leaf":
            print(f'All Values {self.tree[1]}')


        def recursive_rule_finder(tree, rule_starter=None, rules = []):
            class_starter = f'THEN {class_name} ='

            if rule_starter is None:
                rule_starter = f'IF {tree[1]} == '

            for values in tree[2:]:
                value = values[1]
                subtree = values[2]
                #print(f'Value: {value}')

                if subtree[0] == 'Leaf':
                    #print(f'Leafs: {values[2]}')
                    rule = rule_starter + f'{value} {class_starter} {subtree[1]}'
                    rules.append(rule)
                else:
                    new_rule_starter = rule_starter + f'{value} AND {subtree[1]} == '
                    rules = recursive_rule_finder(subtree, new_rule_starter, rules)
            return rules
        
        rules = recursive_rule_finder(self.tree)
        
        filtered_rules = []

        for rule in rules: ##Not a perfect way to filter rules by attribute_names
            for attribute in attribute_names: ##If overlap between attributes names and value names
                if attribute in rule: ##Things will get screwy, good enough for now though
                    filtered_rules.append(rule)
                    break

        # for values in self.tree[2:]:
        #     print(values)

        for rule in filtered_rules:
            print(rule)

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
        dimensions = len(X_test[0])
        distances = []
        neighbor_indices = []

        for point_index in range(len(X_test)):
            top_k = []  # will store (distance, index) for the k nearest neighbors
            for other_point_index in range(len(self.X_train)):
                # Compute squared distance
                distance_between = 0
                for d_index in range(dimensions):
                    distance_between += (X_test[point_index][d_index] - self.X_train[other_point_index][d_index]) ** 2

                if len(top_k) < self.n_neighbors:
                    top_k.append((distance_between, other_point_index))
                else:
                    # Find the max distance in top_k
                    max_dist_index = 0
                    for i, (dist, _) in enumerate(top_k):
                        if dist > top_k[max_dist_index][0]:
                            max_dist_index = i
                    # Replace if the new distance is smaller
                    if distance_between < top_k[max_dist_index][0]:
                        top_k[max_dist_index] = (distance_between, other_point_index)

            # Extract distances and indices and sort them
            top_k.sort(key=lambda x: x[0])
            closest_dist = [d for d, _ in top_k]
            closest_ind = [idx for _, idx in top_k]

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
            class_counter = defaultdict(int)
            for index in group_indices:
                class_counter[self.y_train[index]] += 1

            # pick the class with the highest count
            predicted_class = max(class_counter, key=class_counter.get)
            y_predicted.append(predicted_class)

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
