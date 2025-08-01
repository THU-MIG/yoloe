#!/bin/bash

# Detect architecture
arch=$(uname -m)

if [ "$arch" != "x86_64" ] && [ "$arch" != "aarch64" ]; then
    echo "$arch not supported yet."
    exit 1
fi

TRT_INSTALL_LIBPATH="/usr/lib/$arch-linux-gnu"

# URLs for the different versions of the library
declare -A URLs

# x86_64 URLs
URLs["libnvinfer_plugin.so.8.5.2.x86_64"]="https://github.com/levipereira/deepstream-yolo-e2e/releases/download/v1.0/libnvinfer_plugin.so.8.5.2.x86_64"
URLs["libnvinfer_plugin.so.8.5.3.x86_64"]="https://github.com/levipereira/deepstream-yolo-e2e/releases/download/v1.0/libnvinfer_plugin.so.8.5.3.x86_64"
URLs["libnvinfer_plugin.so.8.6.1.x86_64"]="https://github.com/levipereira/deepstream-yolo-e2e/releases/download/v1.0/libnvinfer_plugin.so.8.6.1.x86_64"
URLs["libnvinfer_plugin.so.8.6.2.x86_64"]="https://github.com/levipereira/deepstream-yolo-e2e/releases/download/v1.0/libnvinfer_plugin.so.8.6.2.x86_64"
URLs["libnvinfer_plugin.so.10.3.0.x86_64"]="https://github.com/levipereira/deepstream-yolo-e2e/releases/download/v1.0/libnvinfer_plugin.so.10.3.0.x86_64"

# aarch64 URLs
URLs["libnvinfer_plugin.so.8.6.1.aarch64"]="https://github.com/levipereira/deepstream-yolo-e2e/releases/download/v1.0/libnvinfer_plugin.so.8.6.1.aarch64"
URLs["libnvinfer_plugin.so.8.6.2.aarch64"]="https://github.com/levipereira/deepstream-yolo-e2e/releases/download/v1.0/libnvinfer_plugin.so.8.6.2.aarch64"
URLs["libnvinfer_plugin.so.10.0.1.aarch64"]="https://github.com/levipereira/deepstream-yolo-e2e/releases/download/v1.0/libnvinfer_plugin.so.10.0.1.aarch64"
URLs["libnvinfer_plugin.so.10.3.0.aarch64"]="https://github.com/levipereira/deepstream-yolo-e2e/releases/download/v1.0/libnvinfer_plugin.so.10.3.0.aarch64"

# Function to download and apply the patch
download_and_patch() {
    local lib_name=$1
    local url=$2

    if [ ! -f "./$lib_name" ]; then
        echo -e "\n\nDownloading: $lib_name"
        curl -L --retry 3 --retry-delay 5 -A "Mozilla/5.0" -o "./$lib_name" "$url" || exit 1
    fi

    # Backup original library if it exists
    if [ -f "$TRT_INSTALL_LIBPATH/$lib_name" ]; then
        mv "$TRT_INSTALL_LIBPATH/$lib_name" "./${lib_name}.ori.bak"
        echo "Backup of the original $lib_name created."
    fi

    # Copy the downloaded library to the installation path
    cp "$(pwd)/$lib_name" "$TRT_INSTALL_LIBPATH/$lib_name"

    # Update library cache
    ldconfig
    echo "$lib_name has been installed successfully."
}

# Function to check and apply the appropriate patch based on architecture
check_and_patch() {
    for lib in "${!URLs[@]}"; do
        if [[ $lib == *"$arch" ]]; then
            lib_base="${lib%.*}"
            if [ -f "$TRT_INSTALL_LIBPATH/$lib_base" ]; then
                download_and_patch "$lib" "${URLs[$lib]}"
                return 0
            fi
        fi
    done
    echo "No compatible version of libnvinfer_plugin.so found in $TRT_INSTALL_LIBPATH."
    exit 1
}

check_and_patch
