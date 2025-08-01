import os
import subprocess
import argparse
import re
import threading
import time
import curses
from prettytable import PrettyTable
import subprocess
from python_module.common.platform_info import PlatformInfo

MODEL_ENGINE_DIR = '/apps/deepstream-yolo-e2e/models/engine/'

# Function to count non-empty lines in the label file
def count_labels(label_file):
    if not os.path.isfile(label_file):
        print(f"Error: Label file '{label_file}' does not exist.")
        return 0
    
    with open(label_file, 'r') as f:
        # Count lines that are not empty or whitespace
        labels = [line.strip() for line in f if line.strip()]
        return len(labels)

# Function to run the trtexec command
def run_trtexec(command):
    """Run the trtexec command without displaying output."""
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Function to get GPU usage using nvidia-smi
def get_gpu_usage():
    """Get GPU 0 usage percentage from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '-i', '0'],  # Specify GPU 0 with '-i 0'
            stdout=subprocess.PIPE, text=True
        )
        gpu_usage = int(result.stdout.strip())  # Convert to integer
        return gpu_usage
    except Exception as e:
        print(f"Error retrieving GPU 0 usage: {e}")
        return 0

def get_gpu_name():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits', '-i', '0'],  # Specify GPU 0
            stdout=subprocess.PIPE, text=True
        )
        gpu_name = result.stdout.strip()
        return gpu_name
    except Exception as e:
        print(f"Error retrieving GPU name: {e}")
        return "Unknown GPU"

def spinner_and_gpu_monitor(stdscr, stop_event, model_name, network_size, batch_size, precision):
    platform_info = PlatformInfo()
    """Display a spinner, GPU usage, and model details in the terminal. Stops when stop_event is set."""
    is_integrated_or_aarch64 = platform_info.is_integrated_gpu() or platform_info.is_platform_aarch64()
    
  
    # Modify model_name to remove everything after the second dash from the end
    modified_model_name = '-'.join(model_name.split('-')[:-2]) if model_name.count('-') > 1 else model_name
    
    # Modify precision to show "int8" if it is "qat"
    modified_precision = "int8" if precision.lower() == "qat" else precision
    
    # Prepare the table with model details
    table = PrettyTable()
    table.field_names = ["Model Name", "Network Size", "Batch Size", "Precision"]
    table.add_row([modified_model_name, network_size, batch_size, modified_precision])
    
    spin_chars = ['-', '\\', '|', '/']
    index = 0

    # Fixed layout that doesn't need to be redrawn each time
    if not is_integrated_or_aarch64:
        stdscr.addstr(1, 0, f"GPU: {get_gpu_name()}")
    stdscr.addstr(3, 0, "Model Information:")
    stdscr.addstr(4, 0, table.get_string())

    # More technical but still futuristic
    stdscr.addstr(10, 0, "Optimizing Neural Network with TensorRT Engines")
    stdscr.addstr(11, 0, "This process may take up to 15 minutes. Please wait.")
    stdscr.addstr(13, 0, f" << Tensor Cores Active >> ")

    while not stop_event.is_set():  # Check if stop_event is set to terminate the loop
        # Get the GPU usage percentage
        gpu_usage = 0
        if not is_integrated_or_aarch64:
            gpu_usage = get_gpu_usage()

        # Create a progress bar for GPU usage
        bar_length = 20
        filled_length = int(bar_length * gpu_usage // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Only update dynamic parts (spinner and GPU usage)
        stdscr.addstr(15, 0, f"Precision Calibration and Layer Fusion in Progress... {spin_chars[index]}")
        if not is_integrated_or_aarch64:
            stdscr.addstr(16, 0, f"GPU Usage: [{bar}] {gpu_usage}%")
        
        # Refresh the screen to update the display
        stdscr.refresh()

        # Update the spinner index
        index = (index + 1) % len(spin_chars)
        time.sleep(0.5)  # Adjust speed of the spinner



def update_output(stdscr, command, model_name, network_size, batch_size, precision ):
    """Curses-based function to manage the spinner and GPU usage."""
    stop_event = threading.Event()  # Create a stop event for the spinner
    
    stdscr.clear()

    # Start the spinner and GPU monitor in a separate thread
    spinner_thread = threading.Thread(target=spinner_and_gpu_monitor, args=(stdscr, stop_event, model_name, network_size, batch_size, precision ))
    spinner_thread.start()

    # Run trtexec in the main thread (no output capture)
    run_trtexec(command)

    # Set stop_event to stop the spinner after trtexec finishes
    stop_event.set()

    # Wait for the spinner thread to finish
    spinner_thread.join()

# Function to process the ONNX file and generate TensorRT engine
def process_onnx(file, label_file, batch_size=1, network_size=640, precision="fp16", pgie_config_file=None, force=False):
    # Check if the file exists
    if not os.path.isfile(file):
        print(f"Error: The file '{file}' does not exist.")
        return

    # Set precision flags
    precision_flags = []
    if precision == "fp32":
        # No flags for fp32
        pass
    elif precision == "fp16":
        precision_flags.append("--fp16")
    elif precision == "qat":
        precision_flags.extend(["--fp16", "--int8"])  # Add both flags for QAT
    else:
        print(f"Error: Invalid precision. Use fp32, fp16, or qat.")
        return

    # Ensure the MODEL_ENGINE_DIR exists
    if not os.path.exists(MODEL_ENGINE_DIR):
        os.makedirs(MODEL_ENGINE_DIR)

    # Extract filename without extension
    filename = os.path.basename(file).replace(".onnx", "")
    model_name = os.path.basename(file).replace(".onnx", "")
    engine_filename = f"{filename}-{precision}-netsize-{network_size}-batch-{batch_size}.engine"
    engine_timing_filename = f"{filename}-{precision}-netsize-{network_size}.engine.timing.cache"

    # Generate full paths for the engine and timing cache files in the MODEL_ENGINE_DIR
    engine_filepath = os.path.join(MODEL_ENGINE_DIR, engine_filename)
    engine_timing_filepath = os.path.join(MODEL_ENGINE_DIR, engine_timing_filename)
    existing_engines = [
        f for f in os.listdir(MODEL_ENGINE_DIR)
        if re.match(rf"{filename}-{precision}-netsize-{network_size}-batch-(\d+)\.engine", f)
    ]
    
    for engine in existing_engines:
        existing_batch_size = int(re.search(r"-batch-(\d+)\.engine", engine).group(1))

        if existing_batch_size >= batch_size:
            engine_filepath = os.path.join(MODEL_ENGINE_DIR, engine)
            break

    if os.path.isfile(engine_filepath) and not force:
        print(f"Warning: The engine file '{engine_filepath}' already exists and will be reused. Use --force to rebuild.")
    else:
        # Run trtexec with the provided options
        command = [
                    "trtexec",
                    f"--onnx={file}",
                    ] + precision_flags + [  # Include precision flags as separate elements
                    f"--saveEngine={engine_filepath}",
                    f"--timingCacheFile={engine_timing_filepath}",
                    "--warmUp=500",
                    "--duration=10",
                    "--useCudaGraph",
                    f"--minShapes=images:1x3x{network_size}x{network_size}",
                    f"--optShapes=images:{batch_size}x3x{network_size}x{network_size}",
                    f"--maxShapes=images:{batch_size}x3x{network_size}x{network_size}"
                ]

        # Start the curses application for the spinner and GPU monitoring
        curses.wrapper(update_output, command, model_name, network_size, batch_size, precision )

    # Count the number of labels (non-empty lines) in the label file
    num_detected_classes = count_labels(label_file)

    # Update configuration file if provided
    if pgie_config_file:
        file_abs_path = os.path.abspath(file)
        engine_abs_path = os.path.abspath(engine_filepath)
        label_file_abs_path = os.path.abspath(label_file)
        update_config_file(pgie_config_file, file_abs_path, engine_abs_path, label_file_abs_path, num_detected_classes, batch_size, network_size)

# Function to update the configuration file
def update_config_file(pgie_config_file, onnx_file, engine_file, label_file, num_detected_classes, batch_size, network_size):
    if os.path.isfile(pgie_config_file):
        with open(pgie_config_file, 'r') as file:
            lines = file.readlines()

        onnx_file_line = f"onnx-file={onnx_file}\n"
        engine_file_line = f"model-engine-file={engine_file}\n"
        label_file_line = f"labelfile-path={label_file}\n"
        batch_size_line = f"batch-size={batch_size}\n"
        infer_dims_line = f"infer-dims=3;{network_size};{network_size}\n"
        num_detected_classes_line = f"num-detected-classes={num_detected_classes}\n"

        updated_lines = []
        for line in lines:
            if line.startswith("onnx-file="):
                updated_lines.append(onnx_file_line)
            elif line.startswith("model-engine-file="):
                updated_lines.append(engine_file_line)
            elif line.startswith("labelfile-path="):
                updated_lines.append(label_file_line)
            elif line.startswith("batch-size="):
                updated_lines.append(batch_size_line)
            elif line.startswith("infer-dims="):
                updated_lines.append(infer_dims_line)
            elif line.startswith("num-detected-classes="):
                updated_lines.append(num_detected_classes_line)
            else:
                updated_lines.append(line)

        with open(pgie_config_file, 'w') as file:
            file.writelines(updated_lines)

        print(f"PGIE Configuration file '{pgie_config_file}' updated.")
    else:
        print(f"Error: PGIE Configuration file '{pgie_config_file}' does not exist.")

# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Process ONNX file and generate TensorRT engine.")
    parser.add_argument("-f", "--file", required=True, help="Name of the .onnx file to be processed")
    parser.add_argument("-l", "--label_file", required=True, help="Path to the label file")
    parser.add_argument("-b", "--batch_size", default=1, type=int, help="Batch size (default: 1)")
    parser.add_argument("-n", "--network_size", default=640, type=int, help="Network size (default: 640)")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "qat"], help="Precision (default: fp16)")
    parser.add_argument("-c", "--pgie_config_file", help="Path to PGIE configuration file")
    parser.add_argument("--force", action="store_true", help="Force re-building the TensorRT engine")

    args = parser.parse_args()

    # Process the ONNX file and generate the engine
    process_onnx(args.file, args.label_file, args.batch_size, args.network_size, args.precision, args.pgie_config_file, args.force)

if __name__ == "__main__":
    main()
