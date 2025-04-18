#!/usr/bin/env python3
"""
Script to generate a submission file for the ARC challenge.
"""

import os
import json
import numpy as np
from pathlib import Path
import argparse

from models.baseline_model import BaselineModel
from utils import load_task, get_all_tasks

def generate_submission(model, test_dir, output_path):
    """
    Generate a submission file for the Kaggle competition.
    
    Args:
        model: Model to use for predictions
        test_dir: Directory containing test tasks
        output_path: Path to save the submission file
    """
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} does not exist. Please download the test data.")
        return
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load all test tasks
    test_tasks = {}
    for filename in os.listdir(test_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(test_dir, filename)
            task_id = os.path.splitext(filename)[0]
            test_tasks[task_id] = load_task(file_path)
    
    if not test_tasks:
        print(f"No test tasks found in {test_dir}.")
        return
    
    print(f"Generating predictions for {len(test_tasks)} tasks...")
    
    # Dictionary to store the submission data
    submission = {}
    
    # For each test task, generate predictions
    for task_id, task in test_tasks.items():
        print(f"Processing task {task_id}...")
        try:
            # Solve the task
            predictions = model.solve_task(task)
            
            # Add the predictions to the submission
            for i, pred in enumerate(predictions):
                # Convert the prediction to a list of lists for JSON serialization
                pred_list = pred.tolist() if isinstance(pred, np.ndarray) else pred
                submission[f"{task_id}_{i}"] = pred_list
        except Exception as e:
            print(f"Error processing task {task_id}: {str(e)}")
            # Add a default prediction (empty grid) for this task
            for i in range(len(task['test'])):
                submission[f"{task_id}_{i}"] = [[]]
    
    # Save the submission to a JSON file
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"Submission saved to {output_path}.")

def main():
    parser = argparse.ArgumentParser(description='Generate a submission file for the ARC challenge.')
    parser.add_argument('--train_dir', type=str, default='data/train', 
                        help='Directory containing training tasks')
    parser.add_argument('--eval_dir', type=str, default='data/evaluation', 
                        help='Directory containing evaluation tasks')
    parser.add_argument('--test_dir', type=str, default='data/test', 
                        help='Directory containing test tasks')
    parser.add_argument('--output', type=str, default='submissions/submission.json', 
                        help='Path to save the submission file')
    args = parser.parse_args()
    
    # Initialize the model
    print("Initializing model...")
    model = BaselineModel()
    
    # Generate submission
    generate_submission(model, args.test_dir, args.output)

if __name__ == "__main__":
    main() 