#!/bin/bash

# Creative Commons Attribution-NonCommercial 4.0 International License
# 
# You are free to share and adapt the material under the following terms:
# - Attribution: Give appropriate credit.
# - NonCommercial: Not for commercial use without permission.
# 
# For inquiries: levi.pereira@gmail.com
# Repository: DeepStream / YOLO (https://github.com/levipereira/deepstream-yolo-e2e)
# License: https://creativecommons.org/licenses/by-nc/4.0/legalcode

# Default values
batch_size=1
network_size=640
precision="fp16"
file=""
config_file=""
force=false

# Function to display usage
usage() {
  echo "Usage: $0 -f file.onnx [-b batch_size] [-n network_size] [-p precision] [-c config_file] [--force]"
  echo "  -f: Name of the .onnx file to be processed"
  echo "  -b: Batch size (default: 1)"
  echo "  -n: Network size (default: 640)"
  echo "  -p: Precision (fp32, fp16, qat; default: fp16)"
  echo "  -c: Configuration PGIE file to update"
  echo "  --force: Force re-generation of the engine file if it already exists"
  exit 1
}

# Parse command line options
while getopts ":f:b:n:p:c:-:" opt; do
  case ${opt} in
    f )
      file=$OPTARG
      ;;
    b )
      batch_size=$OPTARG
      ;;
    n )
      network_size=$OPTARG
      ;;
    p )
      precision=$OPTARG
      ;;
    c )
      config_file=$OPTARG
      ;;
    - )
      case ${OPTARG} in
        force)
          force=true
          ;;
        *)
          usage
          ;;
      esac
      ;;
    \? )
      usage
      ;;
  esac
done
shift $((OPTIND -1))

# Check if the file option is provided
if [ -z "$file" ]; then
  echo "Error: The .onnx file name must be provided."
  usage
fi

# Check if the file exists
if [ ! -f "$file" ]; then
  echo "Error: The file '$file' does not exist."
  exit 1
fi

# Set precision flags
precision_flags=""
case $precision in
  fp32)
    precision_flags=""
    ;;
  fp16)
    precision_flags="--fp16"
    ;;
  qat)
    precision_flags="--fp16 --int8"
    ;;
  *)
    echo "Error: Invalid precision. Use fp32, fp16, or qat."
    usage
    ;;
esac

# Extract directory and filename without extension
file_dir=$(dirname "$file")
filename=$(basename "$file" .onnx)
engine_filename="${filename}-${precision}-netsize-${network_size}-batch-${batch_size}.engine"
engine_timing_filename="${filename}-${precision}-netsize-${network_size}.engine.timing.cache"
# Check if the engine file already exists
engine_timing_filepath="${file_dir}/${engine_timing_filename}"
engine_filepath="${file_dir}/${engine_filename}"
if [ -f "$engine_filepath" ] && ! $force; then
  echo "Warning: The engine file '$engine_filepath' already exists and will be reused. Use --force to rebuild."
else
  # Run trtexec with the provided options
  trtexec \
    --onnx=${file} \
    $precision_flags \
    --saveEngine=${engine_filepath} \
    --timingCacheFile=${engine_timing_filepath} \
    --warmUp=500 \
    --duration=10  \
    --useCudaGraph \
    --minShapes=images:1x3x${network_size}x${network_size} \
    --optShapes=images:${batch_size}x3x${network_size}x${network_size} \
    --maxShapes=images:${batch_size}x3x${network_size}x${network_size}
fi

# Update configuration file if provided
if [ -n "$config_file" ]; then
  if [ -f "$config_file" ]; then
    file_abs_path=$(realpath "$file")
    engine_abs_path=$(realpath "$engine_filepath")
    onnx_file_line="onnx-file=${file_abs_path}"
    engine_file_line="model-engine-file=${engine_abs_path}"
    batch_size_line="batch-size=${batch_size}"
    infer_dims_line="infer-dims=3;${network_size};${network_size}"
    
    sed -i "s|^onnx-file=.*|$onnx_file_line|" "$config_file"
    sed -i "s|^model-engine-file=.*|$engine_file_line|" "$config_file"
    sed -i "s|^batch-size=.*|$batch_size_line|" "$config_file"
    sed -i "s|^infer-dims=.*|$infer_dims_line|" "$config_file"
    echo "Configuration file '$config_file' updated."
  else
    echo "Error: Configuration file '$config_file' does not exist."
    exit 1
  fi
fi
