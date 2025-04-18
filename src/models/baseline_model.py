import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from ..utils.arc_utils import grid_to_features

class BaselineModel:
    """
    A simple baseline model for the ARC challenge that attempts to solve tasks
    using a rule-based approach and a KNN classifier.
    """
    
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=1)
        
    def _flatten_grid(self, grid):
        """Flatten a grid to a 1D array for use in the KNN classifier."""
        grid = np.array(grid)
        # Pad to a standard size if needed (e.g., 30x30)
        padded = np.zeros((30, 30))
        h, w = grid.shape
        padded[:min(h, 30), :min(w, 30)] = grid[:min(h, 30), :min(w, 30)]
        return padded.flatten()
    
    def fit(self, train_examples):
        """
        Train the model on a set of training examples.
        
        Args:
            train_examples: List of dictionaries, each with 'input' and 'output' grids
        """
        X = []
        y = []
        
        for example in train_examples:
            X.append(self._flatten_grid(example['input']))
            y.append(tuple(map(tuple, example['output'])))  # Convert to tuple for KNN
            
        self.knn.fit(X, y)
        
    def predict(self, test_input):
        """
        Predict the output for a test input.
        
        Args:
            test_input: A grid representing the test input
            
        Returns:
            A grid representing the predicted output
        """
        # First, check for some common transformations
        
        # Try identity transformation
        if self._check_identity(test_input, train_examples):
            return test_input
        
        # Try simple color change
        color_change = self._check_color_change(test_input, train_examples)
        if color_change:
            return color_change
        
        # If no simple rule applies, fall back to KNN
        X_test = self._flatten_grid(test_input)
        pred = self.knn.predict([X_test])[0]
        return np.array(pred)
    
    def _check_identity(self, test_input, train_examples):
        """Check if the identity transformation applies."""
        for example in train_examples:
            if np.array_equal(example['input'], example['output']):
                return test_input
        return None
    
    def _check_color_change(self, test_input, train_examples):
        """Check if a simple color change transformation applies."""
        for example in train_examples:
            input_colors = set(np.unique(example['input']))
            output_colors = set(np.unique(example['output']))
            
            # If there's a simple mapping between input and output colors
            if len(input_colors) == len(output_colors):
                # Try to establish a color mapping
                color_map = {}
                for i_color, o_color in zip(sorted(input_colors), sorted(output_colors)):
                    color_map[i_color] = o_color
                
                # Apply the mapping to the test input
                test_output = np.array(test_input).copy()
                for i_color, o_color in color_map.items():
                    test_output[test_output == i_color] = o_color
                    
                return test_output
                
        return None
    
    def solve_task(self, task):
        """
        Solve an ARC task.
        
        Args:
            task: A dictionary containing 'train' and 'test' examples
            
        Returns:
            A list of predicted outputs for each test example
        """
        self.fit(task['train'])
        
        predictions = []
        for test_example in task['test']:
            pred = self.predict(test_example['input'])
            predictions.append(pred)
            
        return predictions 