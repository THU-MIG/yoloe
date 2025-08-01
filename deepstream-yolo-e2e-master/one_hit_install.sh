#!/bin/bash

# Define the DeepStream root directory
DEEPSTREAM_ROOT_DIR="/opt/nvidia/deepstream/"

# Check if the DeepStream root directory exists
if [ ! -d "$DEEPSTREAM_ROOT_DIR" ]; then
    echo "DeepStream is not installed in the directory $DEEPSTREAM_ROOT_DIR"
    exit 1
fi

# Execute the additional DeepStream installation script
bash /opt/nvidia/deepstream/deepstream/user_additional_install.sh
if [ $? -ne 0 ]; then
    echo "Failed to execute additional DeepStream installation script."
    exit 1
fi

# Check the DeepStream version based on the directory name
DEEPSTREAM_VERSION_DIR=$(ls -d ${DEEPSTREAM_ROOT_DIR}/deepstream-* | awk -F'/' '{print $NF}')

# Verify if the version was identified correctly
if [[ $DEEPSTREAM_VERSION_DIR =~ deepstream-([0-9]+\.[0-9]+) ]]; then
    VERSION="${BASH_REMATCH[1]}"
    
    case "$VERSION" in
        "7.1")
            bash /opt/nvidia/deepstream/deepstream/user_deepstream_python_apps_install.sh --version 1.2.0
            ;;
        "7.0")
            bash /opt/nvidia/deepstream/deepstream/user_deepstream_python_apps_install.sh --version 1.1.11
            ;;
        "6.4")
            bash /opt/nvidia/deepstream/deepstream/user_deepstream_python_apps_install.sh --version 1.1.10
            ;;
        "6.3")
            bash /opt/nvidia/deepstream/deepstream/user_deepstream_python_apps_install.sh --version 1.1.8
            ;;
        *)
            echo "Unsupported DeepStream version: $VERSION"
            exit 1
            ;;
    esac

    # Check if the previous command was successful
    if [ $? -ne 0 ]; then
        echo "Failed to install DeepStream Python apps for version $VERSION."
        exit 1
    fi
else
    echo "Unable to identify the DeepStream version in the directory $DEEPSTREAM_ROOT_DIR"
    exit 1
fi

# Compile the nvdsinfer_yolo function used by PGIE
cd /apps/deepstream-yolo-e2e || exit
bash scripts/compile_nvdsinfer_yolo.sh
if [ $? -ne 0 ]; then
    echo "Failed to compile nvdsinfer_yolo function."
    exit 1
fi

# Apply the patch to the nvinfer library
cd /apps/deepstream-yolo-e2e/TensorRTPlugin || exit
bash ./patch_libnvinfer.sh
if [ $? -ne 0 ]; then
    echo "Failed to apply patch to the nvinfer library."
    exit 1
fi

# Install ffmpeg and yt-dlp
apt-get install ffmpeg -y
if [ $? -ne 0 ]; then
    echo "Failed to install ffmpeg."
    exit 1
fi

pip3 install yt-dlp prettytable
if [ $? -ne 0 ]; then
    echo "Failed to install yt-dlp and prettytable."
    exit 1
fi

# Clear the screen at the end
clear
echo "All components have been installed successfully."
