import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def load_task(file_path):
    """
    Load an ARC task from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing the task.
        
    Returns:
        dict: A dictionary containing the task data.
    """
    with open(file_path, 'r') as f:
        task = json.load(f)
    return task

def get_all_tasks(directory):
    """
    Load all tasks from a directory.
    
    Args:
        directory (str): Path to the directory containing the task JSON files.
        
    Returns:
        dict: A dictionary mapping task IDs to task data.
    """
    tasks = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            task_id = os.path.splitext(filename)[0]
            tasks[task_id] = load_task(file_path)
    return tasks

def visualize_grid(grid, title=None):
    """
    Visualize an ARC grid.
    
    Args:
        grid (list): 2D grid to visualize.
        title (str, optional): Title for the plot.
    """
    # ARC grids use integers 0-9 for colors
    cmap = ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', 
                          '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', 
                          '#7FDBFF', '#870C25'])
    
    grid = np.array(grid)
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    plt.grid(True, color='black', linewidth=0.5)
    plt.tick_params(axis='both', which='both', 
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)
    
    # Add grid values as text
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            plt.text(j, i, str(grid[i, j]),
                    ha='center', va='center', 
                    color='white' if grid[i, j] > 0 else 'black')
            
    if title:
        plt.title(title)
    plt.tight_layout()
    
def visualize_task(task):
    """
    Visualize an ARC task with its train and test examples.
    
    Args:
        task (dict): ARC task data.
    """
    train_count = len(task['train'])
    test_count = len(task['test'])
    
    plt.figure(figsize=(12, 4 * (train_count + test_count)))
    
    # Visualize training examples
    for i, example in enumerate(task['train']):
        plt.subplot(train_count + test_count, 2, 2*i + 1)
        visualize_grid(example['input'], f"Train {i+1} Input")
        
        plt.subplot(train_count + test_count, 2, 2*i + 2)
        visualize_grid(example['output'], f"Train {i+1} Output")
    
    # Visualize test examples
    for i, example in enumerate(task['test']):
        plt.subplot(train_count + test_count, 2, 2*(i+train_count) + 1)
        visualize_grid(example['input'], f"Test {i+1} Input")
        
        plt.subplot(train_count + test_count, 2, 2*(i+train_count) + 2)
        visualize_grid(example['output'], f"Test {i+1} Output")
    
    plt.tight_layout()
    plt.show()

def grid_to_features(grid):
    """
    Extract features from an ARC grid.
    
    Args:
        grid (list): 2D grid.
        
    Returns:
        dict: Dictionary of features.
    """
    grid = np.array(grid)
    features = {
        'shape': grid.shape,
        'unique_colors': np.unique(grid).tolist(),
        'color_counts': {int(c): np.sum(grid == c) for c in np.unique(grid)},
        'rows': grid.shape[0],
        'cols': grid.shape[1],
        'total_cells': grid.size,
        'non_zero_cells': np.sum(grid > 0),
    }
    return features 