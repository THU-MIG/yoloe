"""
Creative Commons Attribution-NonCommercial 4.0 International License

You are free to share and adapt the material under the following terms:
- Attribution: Give appropriate credit.
- NonCommercial: Not for commercial use without permission.

For inquiries: levi.pereira@gmail.com
Repository: DeepStream / YOLO (https://github.com/levipereira/deepstream-yolo-e2e)
License: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""

import os
import sys
import json

from python_module.component.manage_sources import manage_source, list_active_media, get_active_sources
from python_module.component.manage_models import choose_model
from python_module.component.onnx_to_trt import process_onnx
from python_module.component.system_config import show_current_resolution, menu_system_resolution
from prettytable import PrettyTable
from python_module.common.utils import display_message



# Define constants for colors (optional)
 
CONFIG_FILE = 'config/python_app/save_session.json'

def load_config():
    """Load the existing configuration from a JSON file."""
    if os.path.isfile(CONFIG_FILE) and os.path.getsize(CONFIG_FILE) > 0:
        with open(CONFIG_FILE, 'r') as file:
            return json.load(file)
    return None

def save_config(config):
    """Save the configuration to a JSON file."""
    with open(CONFIG_FILE, 'w') as file:
        json.dump(config, file)

def prompt_user(question, default='n'):
    """Prompt the user with a question and handle default input."""
    options = f"[Enter to continue or m to modify]" if default == 'n' else "[m to modify or Enter to continue]"
    response = input(f"{question} {options}: ").strip().lower()
    if response == '':  # If the user just presses Enter, return the default value
        return default
    return response

def pre_process(output):
    # Load previous configuration if it exists
    previous_config = load_config()
    if previous_config:
        display_message("d","Previous configuration found.")
        is_media_active = list_active_media()
        
        # Ask if the user wants to modify the media source, default is 'n' (continue without modification)
        if is_media_active:
            modify_media = prompt_user("Do you want to modify the media source?", default='n')

            if modify_media == 'm':
                manage_source()
                while get_active_sources() < 1:
                    manage_source()
                    if get_active_sources() < 1:
                        display_message("w", f"No active media sources were found. \nPlease add or activate a media source before proceeding.\n")
        else:
            manage_source()
            while get_active_sources() < 1:
                manage_source()
                if get_active_sources() < 1:
                    display_message("w", f"No active media sources were found. \nPlease add or activate a media source before proceeding.\n")

        # Ask if the user wants to modify the model, default is 'n' (continue without modification)
        # Create a PrettyTable to display the model configuration
        table = PrettyTable()
        table.field_names = ["Model Type", "Model Name"]
        table.align["Model Type"] = "l"
        table.align["Model Name"] = "l"

        # Add previous configuration values to the table
        num_sources = previous_config.get("num_sources", 0)
        model_file = previous_config.get("model_file", "Not Set")
        label_file = previous_config.get("label_file", "Not Set")
        model_type = previous_config.get("model_type", "Not Set")

        # Define model type for table display
        model_type_table = 'Detection' if model_type == 'det' else 'Segmentation'

        # Extract model name
        base_name = os.path.basename(model_file)
        model_name, _ = os.path.splitext(base_name)
        table.add_row([model_type_table, model_name])

        # Print the table
        display_message("d","\nPrevious Model Configuration:")
        display_message("d",table)

        # Ask if the user wants to modify the model, default is 'n' (continue without modification)
        modify_model = prompt_user("Do you want to modify the model?", default='n')
        if modify_model == 'm':
            model_file, label_file, model_type = choose_model()

    else:
        display_message("w","No previous configuration found. Please configure now.")
        manage_source()
        while get_active_sources() < 1:
            manage_source()
            if get_active_sources() < 1:
                display_message("w", f"No active media sources were found. \nPlease add or activate a media source before proceeding.\n")

        model_file, label_file, model_type = choose_model()

    # Determine model type and configuration file based on user choice
    if model_type in ("Detection", "det"):
        pgie_config_file = 'config/pgie/config_pgie_yolo_det.txt'
        model_type = "det"
    elif model_type in  ("Segmentation","seg"):
        pgie_config_file = 'config/pgie/config_pgie_yolo_seg.txt'
        model_type = "seg"

    # Determine precision based on the model file name
    precision = 'qat' if 'qat' in model_file else 'fp16'
    
    # Save current configurations
    current_config = {
        "num_sources": get_active_sources(),
        "model_file": model_file,
        "label_file": label_file,
        "model_type": model_type,
        "precision": precision,
    }
    save_config(current_config)
    display_message("s","Configuration saved.")

    # Process the ONNX model
    process_onnx(
        file=model_file,
        label_file=label_file,
        batch_size=get_active_sources(),
        network_size=640,
        precision=precision,
        pgie_config_file=pgie_config_file,
        force=False
    )
    if output != "silent":
        show_current_resolution()
        modify_resolution = prompt_user("Do you want to modify current output resolution?", default='n')
        if modify_resolution == 'm':
            menu_system_resolution()
            
    return model_type
