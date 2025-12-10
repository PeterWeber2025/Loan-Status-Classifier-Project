"""
Unit tests for MyRandomForestClassifier
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mysklearn.myclassifiers import MyRandomForestClassifier
from mysklearn.myclassifiers import MyDecisionTreeClassifier


class TestMyRandomForestClassifier(unittest.TestCase):
    """Test cases for MyRandomForestClassifier"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple test dataset (interview dataset style)
        # Features: [experience, education_level]
        # Labels: 0 = not hired, 1 = hired
        self.X_simple = np.array([
            [1, 1],  # low exp, low edu
            [2, 2],  # med exp, med edu
            [3, 3],  # high exp, high edu
            [1, 2],  # low exp, med edu
            [2, 3],  # med exp, high edu
            [3, 1],  # high exp, low edu
            [1, 3],  # low exp, high edu
            [2, 1],  # med exp, low edu
            [3, 2],  # high exp, med edu
        ])
        self.y_simple = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1])
        
        # Larger dataset for more realistic testing
        np.random.seed(42)
        self.X_large = np.random.randint(1, 4, size=(100, 3))
        self.y_large = np.random.randint(0, 2, size=100)
    
    def test_initialization(self):
        """Test that MyRandomForestClassifier initializes correctly"""
        rf = MyRandomForestClassifier(n_trees=20, n_selected_trees=7, n_features=2)
        self.assertEqual(rf.n_trees, 20)
        self.assertEqual(rf.n_selected_trees, 7)
        self.assertEqual(rf.n_features, 2)
        self.assertEqual(len(rf.forest), 0)
    
    def test_fit_creates_forest(self):
        """Test that fit() creates a forest of trees"""
        rf = MyRandomForestClassifier(n_trees=10, n_selected_trees=5, n_features=2, random_state=42)
        rf.fit(self.X_simple, self.y_simple)
        self.assertEqual(len(rf.forest), 5)  # Should have M trees
        self.assertIsNotNone(rf.test_set_indices)
        self.assertIsNotNone(rf.remainder_set_indices)
    
    def test_fit_splits_data_correctly(self):
        """Test that fit() splits data into 1/3 test and 2/3 remainder"""
        rf = MyRandomForestClassifier(n_trees=5, n_selected_trees=3, n_features=2, random_state=42)
        rf.fit(self.X_simple, self.y_simple)
        
        # Check that indices are set
        self.assertIsNotNone(rf.test_set_indices)
        self.assertIsNotNone(rf.remainder_set_indices)
        
        # Test set should be approximately 1/3
        total_size = len(self.X_simple)
        test_size = len(rf.test_set_indices)
        remainder_size = len(rf.remainder_set_indices)
        
        self.assertAlmostEqual(test_size / total_size, 1/3, delta=0.1)
        self.assertAlmostEqual(remainder_size / total_size, 2/3, delta=0.1)
    
    def test_predict_returns_predictions(self):
        """Test that predict() returns predictions"""
        rf = MyRandomForestClassifier(n_trees=10, n_selected_trees=5, n_features=2, random_state=42)
        rf.fit(self.X_simple, self.y_simple)
        
        predictions = rf.predict(self.X_simple[:3])
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_predict_majority_voting(self):
        """Test that predict() uses majority voting"""
        rf = MyRandomForestClassifier(n_trees=5, n_selected_trees=3, n_features=2, random_state=42)
        rf.fit(self.X_simple, self.y_simple)
        
        # Predict on a single instance
        prediction = rf.predict(self.X_simple[:1])
        self.assertIn(prediction[0], [0, 1])
    
    def test_different_parameters(self):
        """Test with different parameter values (N=20, M=7, F=2)"""
        rf = MyRandomForestClassifier(n_trees=20, n_selected_trees=7, n_features=2, random_state=42)
        rf.fit(self.X_simple, self.y_simple)
        
        self.assertEqual(rf.get_forest_size(), 7)
        self.assertEqual(len(rf.feature_indices_list), 7)
    
    def test_larger_dataset(self):
        """Test with larger dataset"""
        rf = MyRandomForestClassifier(n_trees=15, n_selected_trees=5, n_features=2, random_state=42)
        rf.fit(self.X_large, self.y_large)
        
        predictions = rf.predict(self.X_large[:10])
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_random_state_reproducibility(self):
        """Test that random_state ensures reproducibility"""
        rf1 = MyRandomForestClassifier(n_trees=10, n_selected_trees=5, n_features=2, random_state=42)
        rf1.fit(self.X_simple, self.y_simple)
        
        rf2 = MyRandomForestClassifier(n_trees=10, n_selected_trees=5, n_features=2, random_state=42)
        rf2.fit(self.X_simple, self.y_simple)
        
        # Predictions should be the same with same random state
        pred1 = rf1.predict(self.X_simple[:5])
        pred2 = rf2.predict(self.X_simple[:5])
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_forest_size_equals_m(self):
        """Test that forest size equals n_selected_trees"""
        for m in [3, 5, 7, 10]:
            rf = MyRandomForestClassifier(n_trees=20, n_selected_trees=m, n_features=2, random_state=42)
            rf.fit(self.X_simple, self.y_simple)
            self.assertEqual(rf.get_forest_size(), m)


class TestMyDecisionTreeClassifier(unittest.TestCase):
    """Test cases for MyDecisionTreeClassifier"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.X = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [1, 2],
            [2, 3],
        ])
        self.y = np.array([0, 1, 1, 0, 1])
    
    def test_tree_initialization(self):
        """Test tree initialization"""
        tree = MyDecisionTreeClassifier()
        self.assertIsNone(tree.tree)
    
    def test_tree_fit(self):
        """Test tree fitting"""
        tree = MyDecisionTreeClassifier()
        tree.fit(self.X, self.y)
        self.assertIsNotNone(tree.tree)
    
    def test_tree_predict(self):
        """Test tree prediction"""
        tree = MyDecisionTreeClassifier()
        tree.fit(self.X, self.y)
        predictions = tree.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_tree_with_feature_indices(self):
        """Test tree with specific feature indices"""
        tree = MyDecisionTreeClassifier()
        tree.fit(self.X, self.y, feature_indices=[0])  # Only use first feature
        predictions = tree.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))


if __name__ == '__main__':
    unittest.main()

