"""
Creative Commons Attribution-NonCommercial 4.0 International License

You are free to share and adapt the material under the following terms:
- Attribution: Give appropriate credit.
- NonCommercial: Not for commercial use without permission.

For inquiries: levi.pereira@gmail.com
Repository: DeepStream / YOLO (https://github.com/levipereira/deepstream-yolo-e2e)
License: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""

import json
import os
import sys
import requests
from prettytable import PrettyTable

MODEL_ASSET = 'config/models/models_asset.json'
MODEL_CATALOG = 'config/models/models_catalog.json'
MODEL_ONNX_DIR = 'models/onnx/'

# Load model structure from the JSON file
def load_models():
    if not os.path.exists(MODEL_CATALOG):
        print(f"Error: File not found at {MODEL_CATALOG}")
        return {}
    
    with open(MODEL_CATALOG, 'r') as f:
        return json.load(f)

def download_file(url, name):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Get the total file size from the response headers
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        # Save the file and show progress
        with open(name, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive new chunks
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    # Calculate progress
                    progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                    # Print progress
                    sys.stdout.write(f"\rDownloading {name}: {progress:.2f}%")
                    sys.stdout.flush()

        print(f"\nDownloaded: {name}")
    except Exception as e:
        print(f"Error downloading {name}: {e}")

def download_model(model_name):
    # Load asset information from JSON
    with open(MODEL_ASSET, 'r') as file:
        data = json.load(file)

    # Initialize a list to store URLs for downloading
    download_urls = []

    # Search for the model files in the asset data
    for asset in data["assets"]:
        if model_name in asset['name']:  # Check if the model_name is in the asset's name
            download_urls.append((asset['url'], asset['name']))  # Collect the download URL and file name

    # Check if any files were found for the model
    if not download_urls:
        print(f"No files found for model {model_name}.")
        return None, None

    # Ensure the download directory exists
    os.makedirs(MODEL_ONNX_DIR, exist_ok=True)

    # Download each file found
    for url, file_name in download_urls:
        file_path = os.path.join(MODEL_ONNX_DIR, file_name)
        
        # Check if file already exists
        if os.path.exists(file_path):
            print(f"{file_name} already exists, skipping download.")
        else:
            download_file(url, file_path)  # Download the file

    # Return paths to the downloaded ONNX and label files
    model_file = os.path.join(MODEL_ONNX_DIR, f"{model_name}.onnx")
    label_file = os.path.join(MODEL_ONNX_DIR, f"{model_name}.txt")
    return model_file, label_file

def display_table(data, title):
    table = PrettyTable()
    table.field_names = ["Index", "Option"]
    
    # Align text to the right
    table.align["Option"] = "l"
    table.align["Index"] = "c"  # Center align the index column
    
    for index, option in enumerate(data, start=1):
        table.add_row([index, option])
    
    print(f"\n{title}")
    print(table)

def choose_model():
    models = load_models()
    while True:
        # Show datasets with descriptions
        dataset_data = [(dataset, models[dataset]["Description"]) for dataset in models.keys()]

        # Create table for datasets
        dataset_table = PrettyTable()
        dataset_table.field_names = ["Index", "Dataset", "Description"]
        
        # Align text to the right
        dataset_table.align["Dataset"] = "l"
        dataset_table.align["Description"] = "l"
        dataset_table.align["Index"] = "c"  # Center align the index column
        
        for index, (dataset, description) in enumerate(dataset_data, start=1):
            dataset_table.add_row([index, dataset, description])
        
        print("\nChoose a Dataset (or enter '0' to exit):")
        print(dataset_table)

        dataset_choice = input("Enter your choice: ")
        if dataset_choice == '0':
            print("Exiting the application.")
            return None, None, None
        
        try:
            dataset_index = int(dataset_choice) - 1
            dataset = list(models.keys())[dataset_index]
            print(f"\nYou chose: {dataset}")
        except (ValueError, IndexError):
            print("Invalid choice, please try again.")
            continue

        # Show model types based on dataset
        model_types = ["Detection", "Segmentation"] if dataset == "COCO" else ["Detection"]
        display_table(model_types, "Choose Model Type:")
        model_type_choice = input("Enter your choice (or '0' to go back): ")
        if model_type_choice == '0':
            continue
        
        try:
            model_type_index = int(model_type_choice) - 1
            model_type = model_types[model_type_index]
            print(f"\nYou chose: {model_type}")
        except (ValueError, IndexError):
            print("Invalid choice, please try again.")
            continue

        # Show sizes based on model type
        sizes = list(models[dataset][model_type].keys())
        display_table(sizes, "Choose Size:")
        size_choice = input("Enter your choice (or '0' to go back): ")
        if size_choice == '0':
            continue
        
        try:
            size_index = int(size_choice) - 1
            size = sizes[size_index]
            print(f"\nYou chose: {size}")
        except (ValueError, IndexError):
            print("Invalid choice, please try again.")
            continue

        # Show available models with architecture and model_name
        available_models = models[dataset][model_type][size]
        model_data = [(model['model_arch'], model['model_name']) for model in available_models]

        # Display models in a formatted table
        table = PrettyTable()
        table.field_names = ["Index", "Model Arch", "Model Name"]
        
        # Align text to the right
        table.align["Model Arch"] = "l"
        table.align["Model Name"] = "l"
        table.align["Index"] = "c"  # Center align the index column
        
        for index, (model_arch, model_name) in enumerate(model_data, start=1):
            table.add_row([index, model_arch, model_name])
        
        print("\nAvailable Models:")
        print(table)

        # Model selection
        model_index = input("\nEnter the index of the model you want to download (or '0' to go back): ")
        if model_index == '0':
            continue

        try:
            model_index = int(model_index) - 1
            if 0 <= model_index < len(model_data):
                selected_model = model_data[model_index][1]  # Get model_name
                model_file, label_file = download_model(selected_model)
                if model_file and label_file:
                    print(f"Model downloaded to: {model_file}")
                    print(f"Label file downloaded to: {label_file}")
                print("Exiting the application after downloading the model.")
                return model_file, label_file, model_type  # Return model file, label file, and model type
            else:
                print("Invalid model choice!")
        except ValueError:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    
    model_file, label_file, model_type = choose_model()
    print(f"Model File: {model_file}")
    print(f"Label File: {label_file}")
    print(f"Model Type: {model_type}")