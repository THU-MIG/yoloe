#!/usr/bin/python3
"""
Creative Commons Attribution-NonCommercial 4.0 International License

You are free to share and adapt the material under the following terms:
- Attribution: Give appropriate credit.
- NonCommercial: Not for commercial use without permission.

For inquiries: levi.pereira@gmail.com
Repository: DeepStream / YOLO (https://github.com/levipereira/deepstream-yolo-e2e)
License: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""

import sys
import argparse
from prettytable import PrettyTable
from python_module.component.pipeline import run_pipeline
from python_module.common.utils import clear_screen
import os

os.environ['GST_DEBUG'] = 'ERROR'  
os.system('stty sane')

def display_output_options():
    """Display available output options to the user in a table format."""
    table = PrettyTable()
    table.field_names = ["Index", "Output Option", "Description"]
    table.align["Index"] = "l"
    table.align["Output Option"] = "l"
    table.align["Description"] = "l"

    options = [
        ("1", "display", "Output to display window"),
        ("2", "file", "Save output to a file"),
        ("3", "rtsp", "Stream output via RTSP"),
        ("4", "silent", "No output"),
    ]
    
    for option in options:
        table.add_row(option)

    print("Please choose the output option:")
    print(table)

def get_user_choice():
    """Get the user's choice for output option."""
    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if choice in range(1, 5):  # Valid choices are 1-4
                return ["display", "file", "rtsp", "silent"][choice - 1]
            else:
                print("Invalid choice. Please select a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nInput interrupted. Exiting application.")
            sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(prog="pipeline_yolo.py",
                                     description="pipeline_yolo multi stream, multi model inference reference app")
    
    # Adding the output argument as optional
    parser.add_argument(
        "-o",
        "--output",
        help="Output",
        choices=["display", "file", "rtsp", "silent"],
    )

    # Parse arguments
    args = parser.parse_args()

    # If the user provided an output option via the command line, use it.
    if args.output:
        return args

    # Otherwise, prompt the user for the output option.
    display_output_options()
    selected_output = get_user_choice()
    return argparse.Namespace(output=selected_output)

if __name__ == '__main__':
    clear_screen()
    try:
        args = parse_args()
        sys.exit(run_pipeline(args))
    except KeyboardInterrupt:
        os.system('stty sane')
        print("\nApplication terminated by user.")
        sys.exit(0)
