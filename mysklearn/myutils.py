'''
# Programmer: Peter Weber
# 11/29/25
# Description: An assortment of functions I've been building up
'''
from tabulate import tabulate
from mysklearn import myevaluation

from tabulate import tabulate
from mysklearn import myevaluation


def label_encoder(label, list_labels, list_values):
    '''
    Expects That the the list of values to substitute is parallel to this list of
    values to subsitute with i.e: [Gold,Silver,Bronze] [1,2,3]
    '''
    if label in list_labels:
        label_index = list_labels.index(label)

        return list_values[label_index]
    
    return label



def count_label_distribution(list_1d):
    label_counter = {}

    for value in list_1d:
        label_counter[value] = label_counter.get(value, 0) + 1
    
    return label_counter



def find_attribute_domains(instances, available_attributes, header):
    attribute_domains = {}

    for attribute in available_attributes:
        attribute_domains[attribute] = []
        attribute_index = header.index(attribute)

        column_values = [instances[row_index][attribute_index] for row_index in range(len(instances))]
        
        unique_column_values = list(set(column_values))
        unique_column_values.sort()
        attribute_domains[attribute] = unique_column_values
    #print(attribute_domains)
    return attribute_domains

def all_same_class(instances):
    first_class = instances[0][-1] ## [-1] because y is always last index in row
    for instance in instances:
        # if any label differs, return False
        if instance[-1] != first_class:
            return False
        
    # if the loop completes without finding differences, return True.
    return True 


def formatted_confusion_matrix(confusion_matrix, top_left_corner, header):

    for row_index in range(len(confusion_matrix)):
        if sum(confusion_matrix[row_index]) > 0:
            recog_rate = round(confusion_matrix[row_index][row_index] / sum(confusion_matrix[row_index]),2)
        else:
            recog_rate = 0.0
            
        confusion_matrix[row_index].append(sum(confusion_matrix[row_index]))
        confusion_matrix[row_index].insert(0, header[row_index])
        confusion_matrix[row_index].append(recog_rate)
        
    header.insert(0, top_left_corner)
    header.append('Total')
    header.append('Recognition %')
    print(tabulate(confusion_matrix, headers=header))

def precision_recall_f1_printer(y_true, y_pred, labels=None, pos_label=None):
    precision = round(myevaluation.binary_precision_score(y_true, y_pred, labels, pos_label),2)
    recall = round(myevaluation.binary_recall_score(y_true, y_pred, labels, pos_label),2)
    f1_score = round(myevaluation.binary_f1_score(y_true, y_pred, labels, pos_label),2)

    print(f'The classifier had a precision of: {precision}, a recall of: {recall}, and f1_score of: {f1_score}')

def accuracy_error_printer(y_true, y_pred):
    accuracy = myevaluation.accuracy_score(y_true,y_pred)

    print(f'The classifier has an accuracy of: {round(accuracy,2)}, and an error rate of: {round(1-accuracy,2)}')

def k_fold_predictions(classifier_object,data_to_fold, k_count):
    fold_tuples = myevaluation.kfold_split(data_to_fold, k_count)

    y_predictions = []

    for tuple in fold_tuples:
        train_data = tuple[0]
        test_data = tuple[1]

        x_test = []
        x_train = []
        y_train = []

        for index in train_data:
            x_train.append(data_to_fold[index][:-1])
            y_train.append(data_to_fold[index][-1])
        for index in test_data:
            x_test.append(data_to_fold[index][:-1])

        classifier_object.fit(x_train, y_train)
        y_predictions.extend(classifier_object.predict(x_test))

    return y_predictions


def bootstrap_method(classifier_object, X, y, test_size=.33, k_count=5):

    accuracy_list = []

    for _ in range(k_count):
        
        x_train, x_test, y_train, y_test = myevaluation.bootstrap_sample(X,y)
        
        classifier_object.fit(x_train, y_train)

        y_predict = classifier_object.predict(x_test)

        accuracy = myevaluation.accuracy_score(y_test, y_predict)

        accuracy_list.append(accuracy)
    
    mean_accuracy = round(sum(accuracy_list) / len(accuracy_list),2)

    return mean_accuracy




def cross_val_predict(classifier_object, unfolded_data, n_splits,shuffle=False):
    '''
    Assumes Y Value Is Last Column Of Data
    '''
    
    accuracy_list = []

    fold_tuples = myevaluation.kfold_split(unfolded_data, n_splits,shuffle=shuffle)

    for tuple in fold_tuples:
        train_data = tuple[0]
        test_data = tuple[1]
            
        x_test = []
        y_test = []
        x_train = []
        y_train = []

        for index in train_data:
            x_train.append(unfolded_data[index][:-1])
            y_train.append(unfolded_data[index][-1])

        for index in test_data:
            x_test.append(unfolded_data[index][:-1])
            y_test.append(unfolded_data[index][-1])


        classifier_object.fit(x_train, y_train)

        y_predict = classifier_object.predict(x_test)

        accuracy = myevaluation.accuracy_score(y_test, y_predict)

        accuracy_list.append(accuracy)
    
    mean_accuracy = round(sum(accuracy_list) / len(accuracy_list),2)


    return mean_accuracy


def random_subsample(classifier_object, X, y, test_size=.33, k_count=5):

    accuracy_list = []

    for _ in range(k_count):
        x_train, x_test, y_train, y_test = myevaluation.train_test_split(X,y,test_size)
        
        classifier_object.fit(x_train, y_train)

        y_predict = classifier_object.predict(x_test)

        accuracy = myevaluation.accuracy_score(y_test, y_predict)

        accuracy_list.append(accuracy)
    
    mean_accuracy = round(sum(accuracy_list) / len(accuracy_list),2)

    return mean_accuracy



def mpg_ranking_discretizer(value):
    if value >= 45:
        return 10
    elif value >= 15:
        rating = 9
        fuel_economy_rating_key = [37, 31, 27, 24, 20, 17, 15]
        for threshold in fuel_economy_rating_key:
            if value >= threshold:
                return rating
            else:
                rating -= 1
    elif value == 14:
        return 2
    else:
        return 1


def print_attribute_index(list_1d, attributes):    
    for attribute in attributes:
        print(f'{attribute} Index: {list_1d.index(attribute)}')


def table_normalizer(list_2d):
    ##Assumes Columns Have Independent Data
    num_cols = len(list_2d[0])
    num_rows = len(list_2d)

    normalized_table = [[-1] * num_cols for _ in range(num_rows)]

    for col in range(num_cols):
        # Extract that column
        column_values = [list_2d[row][col] for row in range(num_rows)]

        col_min = min(column_values)
        col_max = max(column_values)
        
        
        denom = col_max - col_min

        if denom == 0:
            # All values are the same; assign 0 to normalized column
            for row in range(num_rows):
                normalized_table[row][col] = 0.0

        for row in range(num_rows):
            normalized_table[row][col] = (list_2d[row][col] - col_min) / denom

    return normalized_table


def my_discretizer(y_val):
    if y_val >= 100:
        return 'high'
    return 'low'

def header_formatter(header, length=1):
    string_length = len(str(header))
    num_dashes = '=' * string_length * length
    print(num_dashes)
    print(header)
    print(num_dashes)
    pass


def find_item_in_table(value_to_find, table, index_start=0, index_stop=None, require_all=True):
    """
    Finds Rows In 2D List That Contain Specific Value
    if all == False, will find rows where any index between start and stop is the value
    if all == True, will find rows where every index between start and stop is the value

    Returns:
        2D list of rows containing specified Value
    """
    rows_with_value = []

    slicer = slice(index_start, index_stop)

    if require_all:
        for row in table:
            if all(value_to_find == item for item in row[slicer]): ## This is just a way of checking that every item in row is 'NA'
                #print(row)
                rows_with_value.append(row)
    else:
        for row in table:
            for item in row[slicer]:
                if value_to_find == item:
                    rows_with_value.append(row)
                    break

    return rows_with_value

                    
def find_bin_cutoffs(list_1d):
    list_1d.sort()

    bins = 5

    minimum = list_1d[0]
    maximum = list_1d[-1]

    mpg_range  = maximum - minimum
    width = mpg_range / bins
    #print(width)

    bin_cutoffs = [minimum + i*width for i in range(bins)]
    bin_cutoffs.append(maximum)

    return bin_cutoffs

def sort_values_into_bins(list_1d, cutoffs):
    bins_dict = {}

    for value in range(len(cutoffs) - 1): ##Assuming we always have 1 more cutoff than bin
         bins_dict[value + 1] = 0

    for value in list_1d:
        if value == cutoffs[-1]: ##We do this to avoid an out of bounds error in the below loop
            bins_dict[len(cutoffs) - 1] += 1
            continue
        for threshold_index in range(len(cutoffs)):
            if value < cutoffs[threshold_index + 1]:
                bins_dict[threshold_index + 1] += 1
                break
    return bins_dict
