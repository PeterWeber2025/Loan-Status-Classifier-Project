
'''
# Programmer: Peter Weber
# Class: Cpsc 322-01, Fall 2025
# Programming Assignment #4
# 11/23/25
# Description: A class that mimics (very) partial functionality of pandas 
'''

import copy
import csv
from tabulate import tabulate
#from mysklearn import myutils

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        row_length = len(self.data)
        column_length = len(self.column_names)
        return row_length, column_length

    def print_shape(self, columns=False,row=False,shape=True):
        """Prints the dimension of the table (N x M)"""

        if columns:
            print(f'Column Names: {self.column_names}')

        if row:
            print(f'First Row: {self.data[0]}')

        if shape:
            table_shape = self.get_shape()
            print(f'The Table Has {table_shape[1]} Attributes And {table_shape[0]} instances')

    def get_column(self, col_identifier, include_missing_values=True): ##Check With NA values
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        if type(col_identifier) == int:
            if col_identifier in range(len(self.column_names) - 1) and col_identifier >= 0:
                col_index = col_identifier
            else:
                raise ValueError

        elif type(col_identifier) == str:
            try:
                col_index = self.column_names.index(col_identifier)
            except Exception as exc:
                print(f"{col_identifier} not in {self.column_names}")
                raise ValueError from exc

        column_list = []

        if include_missing_values is False:
            for row in self.data:
                if row[col_index] != 'NA' and row[col_index] != '':
                    column_list.append(row[col_index])
        else:
            for row in self.data:
                column_list.append(row[col_index])
        return column_list

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float or int).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        failed_to_modify = 0

        for row_index in range(len(self.data)):
            for item_index in range(len(self.data[row_index])):
                try:
                    item_value = float(self.data[row_index][item_index])
                    if item_value.is_integer():
                        item_value = int(item_value)
                    self.data[row_index][item_index] = item_value
                except:
                    failed_to_modify += 1
                    #print(type(self.data[row_index][item_index]))

        #print(failed_to_modify)


    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        for index in sorted(row_indexes_to_drop, reverse=True): ## Sorting so I don't mess up subsequent removals
            self.data.pop(index)
    
    def drop_column(self, attribute):
        if attribute not in self.column_names:
            print(f'Attribute {attribute} Missing From Column Names')
            return 0 
        attribute_index = self.column_names.index(attribute)
        self.column_names.pop(attribute_index)
        for row in self.data:
            row.pop(attribute_index)


    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        data_table = []
        with open(filename, encoding="utf-8") as infile:
            dataset = csv.reader(infile)
            for line in dataset:
                data_table.append(line)

        self.column_names = data_table.pop(0)
        self.data = data_table
        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """

        if '.csv' not in filename and '.txt' not in filename:
            print('missing .csv or .txt at end of filepath, added .csv manually to end of string')
            filename = filename + '.csv'

        with open(filename, 'w', newline='', encoding="utf-8") as csv_file: ##Looked up the CSV module: https://docs.python.org/3/library/csv.html
            writer = csv.writer(csv_file)
            writer.writerow(self.column_names)
            for row in self.data:
                writer.writerow(row)


    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """

        keys_index_list = []
        keys_list = []
        duplicates_index = []

        for key in key_column_names:
            keys_index_list.append(self.column_names.index(key))

##Switched to enumerate because self.data.index(row) would only ever return
##the index of the first duplicate (there could me multiple)'''

        for row_index, row in enumerate(self.data):
            key = ''
            for key_index in keys_index_list:
                key = key + '/' + str(row[key_index])

            if key in keys_list:
                duplicates_index.append(row_index)

            else:
                keys_list.append(key)

        #print(duplicates_index)
        duplicates_index.sort(reverse=True)
        return duplicates_index

    def duplicate_printer(self, key_column_names, print_rows=True):
        '''Prints out indices of repeated values in table'''

        duplicate_indices = self.find_duplicates(key_column_names)
        print(f'Number Of Duplicates: {len(duplicate_indices)}')

        if print_rows is True:
            for i in duplicate_indices:
                print(self.data[i])
        return duplicate_indices

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""

        indices_to_drop = []

        for row_index, row in enumerate(self.data):
            for item in row:
                if item == 'NA' or item == '' or item == 'n/a':
                    indices_to_drop.append(row_index)
                    break

        # for index in indices_to_drop:
        #     print(self.data[index])

        self.drop_rows(indices_to_drop)
    
    def print_nan_by_column(self, null_print=False):
        nan_attribute_counter = {}

        for attribute in self.column_names:
            nan_attribute_counter[attribute] = 0
            attribute_column = self.get_column(attribute)
            for value in attribute_column:
                if value == 'NA' or value == '' or value == 'n/a':
                    nan_attribute_counter[attribute] += 1
        
        if null_print:
            print(nan_attribute_counter)
        return nan_attribute_counter

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """

        ###Part 1 Make Sure Column Has continuous data
        col_index = self.column_names.index(col_name)

        if type(self.data[0][col_index]) != float and type(self.data[0][col_index]) != int:
            print(f"Warning the attribute {col_name} is of type: {type(self.data[0][col_index])}")
            print('Aborting Operation') ###Strange phrasing I know, but didn't want to be to wordy

        ###Part 2, Finding Column Average
        else:
            column_list = self.get_column(col_name, False)
            average = round(sum(column_list) / len(column_list), 2)

            #print(average)

        ###Part3, Replacing Values
            for row_index, row in enumerate(self.data):
                if row[col_index] == 'NA':
                    self.data[row_index][col_index] = average
                    #print(self.data[row_index])


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        summary_table = MyPyTable()
        summary_table.column_names = ["attribute", "min", "max", "mid", "avg", "median"]

        for column in col_names:
            column_list = sorted(self.get_column(column, False))

            if column_list:
                column_length = len(column_list)
                min_value = min(column_list)
                max_value = max(column_list)
                mid_range = (max_value + min_value) / 2
                average = sum(column_list) / column_length

                if column_length % 2 == 0:
                    medianindex1 = column_length//2
                    medianindex2 = medianindex1 - 1
                    median = (column_list[medianindex1] + column_list[medianindex2]) / 2
                else:
                    median = column_list[column_length//2]

                row = [column, min_value, max_value, mid_range, average, median]
                summary_table.data.append(row)

        return summary_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """

        if set(self.column_names) >= set(key_column_names) and set(other_table.column_names) >= set(key_column_names):

            ##Part 1, Set Column Names
            inner_join_table = MyPyTable()
            merged_column_names = self.column_names[:]

            for column in other_table.column_names:
                if column not in merged_column_names:
                    merged_column_names.append(column)


            inner_join_table.column_names = merged_column_names
            ##Part 2, Find Key Column Name Indexes For Both Tables, Figure Out Which Attributes From Other_Table, Aren't In Self_Table
            self_key_index = []
            other_key_index = []
            indices_for_other_only_attributes = []
            for key in key_column_names:
                self_key_index.append(self.column_names.index(key))
                other_key_index.append(other_table.column_names.index(key))

            for column in other_table.column_names:
                if column not in self.column_names:
###Because These Are Added In The Same Order As They Were To merged_column_names, Shouldn't be any problems
                    indices_for_other_only_attributes.append(other_table.column_names.index(column))

            ##Part 3, Match On Keys and append to data
            for self_row in self.data:
                self_key = ''
                for index in self_key_index:
                    self_key = self_key + '/' + str(self_row[index]) ##For each row in self.data, find it's key

                for other_row in other_table.data:
                    other_key = ''
                    for index in other_key_index:
                        other_key = other_key + '/' + str(other_row[index])
                    if self_key == other_key: ##Check The Key For Each self.data row, against every row in other_table.data
                        record = self_row[:]
                        for index in indices_for_other_only_attributes:
                            record.append(other_row[index])
                        inner_join_table.data.append(record)
            return inner_join_table
        print('Key Columns Not Found In Both Tables')


    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """

        if set(self.column_names) >= set(key_column_names) and set(other_table.column_names) >= set(key_column_names):

            ##Part 1, Set Column Names
            indices_for_other_only_attributes = []
            merged_column_names = self.column_names[:]

            for column in other_table.column_names:
                if column not in merged_column_names:
                    merged_column_names.append(column)
                    indices_for_other_only_attributes.append(other_table.column_names.index(column))

            outer_join_table = MyPyTable(merged_column_names)

            ##Part 2, Find Key Column Name Indexes For Both Tables, Figure Out Which Attributes From Other_Table, Aren't In Self_Table
            self_key_index = []
            other_key_index = []

            for key in key_column_names:
                self_key_index.append(self.column_names.index(key))
                other_key_index.append(other_table.column_names.index(key))

            ##Part 3, Match When Possible Otherwise Append With NA
            ##We need to record all the data not just matches, so we can't just iterate through all the data from one table
            ##Need Tables To Figure Out What's Already Been Matched
            matched_self_indices = set()
            matched_other_indices = set( )




            for self_index, self_row in enumerate(self.data):
                self_key = ''
                for index in self_key_index: #A loop that gets the key for each row of self
                    self_key = self_key + '/' + str(self_row[index])


                for other_index, other_row in enumerate(other_table.data):
                    other_key = ''
                    for index in other_key_index: #Same as above, but for other
                        other_key = other_key + '/' + str(other_row[index])


                    if self_key == other_key: ##If The Two Keys are Matches . . .
                        matched_self_indices.add(self_index)
                        matched_other_indices.add(other_index)
                        record = self_row[:]

                        for index in indices_for_other_only_attributes:
                            record.append(other_row[index])
                        outer_join_table.data.append(record)

            ##With All The Matches Complete, Whats Left Is To Add Rows With 'NA's and either self or other table data

            for self_index, row in enumerate(self.data):
                if self_index not in matched_self_indices:
                    record = []
                    for column in outer_join_table.column_names:
                        if column in self.column_names:
                            record.append(row[self.column_names.index(column)])
                        else:
                            record.append('NA')
                    outer_join_table.data.append(record)

            for other_index, row in enumerate(other_table.data):
                if other_index not in matched_other_indices:
                    record = []
                    for column in outer_join_table.column_names:
                        if column in other_table.column_names:
                            record.append(row[other_table.column_names.index(column)])
                        else:
                            record.append('NA')
                    outer_join_table.data.append(record)

            return outer_join_table
        else:
            print('Key Columns Not Found In Both Tables')


    def frequency_counter(self, column_name):
        """Return a dictionary with keys that are every unique instance in a column
        and values that are the number of times each instance occurs in the column

        Parameters:
            column_name(str): column name to find frequency for.

        Returns:
            Dictionary with value frequency
        """

        counter_dict = {}

        column = self.get_column(column_name)

        for item in column:
            if item in counter_dict.keys():
                counter_dict[item] += 1
            else:
                counter_dict[item] = 1
        return counter_dict
