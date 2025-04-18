#!/usr/bin/env python3
"""
Script to download the ARC dataset from the official GitHub repository.
"""

import os
import json
import urllib.request
import shutil
import zipfile
from pathlib import Path
import argparse

def download_data(output_dir, force=False):
    """
    Download the ARC dataset from the official GitHub repository.
    
    Args:
        output_dir: Directory to save the dataset
        force: If True, download even if the data already exists
    """
    base_url = "https://github.com/fchollet/ARC/raw/master/data"
    data_types = {
        "training": "training.zip",
        "evaluation": "evaluation.zip"
    }
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for data_type, file_name in data_types.items():
        # Create the data type directory if it doesn't exist
        data_type_dir = os.path.join(output_dir, data_type.replace("ing", ""))
        os.makedirs(data_type_dir, exist_ok=True)
        
        # Skip if the directory already has data and force is False
        if not force and len(os.listdir(data_type_dir)) > 0:
            print(f"{data_type.capitalize()} data already exists at {data_type_dir}. Use --force to redownload.")
            continue
        
        # Download the zip file
        url = f"{base_url}/{file_name}"
        zip_path = os.path.join(output_dir, file_name)
        
        print(f"Downloading {data_type} data from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"Error downloading {data_type} data: {str(e)}")
            continue
        
        # Extract the zip file
        print(f"Extracting {data_type} data to {data_type_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_type_dir)
        
        # Remove the zip file
        os.remove(zip_path)
        
        print(f"{data_type.capitalize()} data downloaded and extracted to {data_type_dir}.")

def main():
    parser = argparse.ArgumentParser(description='Download the ARC dataset.')
    parser.add_argument('--output_dir', type=str, default='data', 
                        help='Directory to save the dataset')
    parser.add_argument('--force', action='store_true', 
                        help='Download even if the data already exists')
    args = parser.parse_args()
    
    download_data(args.output_dir, args.force)

if __name__ == "__main__":
    main() 