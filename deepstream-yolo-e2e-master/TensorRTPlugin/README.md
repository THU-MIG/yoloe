# Build Process TensorRT Custom plugin

TensorRT OSS build is required. This is required because TensorRT TRT_EfficientNMSX plugin that are required by models  and not in the general TensorRT release. 

## New plugin
- [TRT_EfficientNMSX](https://github.com/levipereira/TensorRT/tree/release/8.6/plugin/efficientNMSPlugin): Same Efficient NMS, but return boxes indices



## Setting Up The Build Environment

For Linux platforms, we recommend that you generate a docker container for building TensorRT OSS as described below. 

## DEEPSTREAM 7.1
1. #### Download TensorRT OSS - DeepStream 7.0
	```bash
    cd TensorRTPlugin
	git clone -b release/10.3_efficientnmsx https://github.com/levipereira/TensorRT TensorRT
	cd TensorRT
	git submodule update --init --recursive
	```
2. #### Generate the TensorRT-OSS build container. 
    The TensorRT-OSS build container can be generated using the supplied Dockerfiles and build scripts. The build containers are configured for building TensorRT OSS out-of-the-box.

    **Ubuntu 22.04 on x86-64 with cuda-12.6.2/cudnn9 for DeepStream 7.1** 
    ```bash
    ./docker/build.sh --file docker/ubuntu-22.04_deepstream_7.1.Dockerfile --tag tensorrt-ubuntu22.04-deepstream71-cuda12.6  
    ```

3. #### Launch the TensorRT-OSS build container.
    **Example: Ubuntu 20.04 build container**
	```bash
	./docker/launch.sh --tag tensorrt-ubuntu22.04-deepstream71-cuda12.6  --gpus all
	```
	> NOTE:
  <br> `sudo` password for Ubuntu build containers is `nvidia`.
## Building TensorRT-OSS
* Generate Makefiles and build.

    **Example: Linux (x86-64) build with default cuda-12.1 - DeepStream 7.0**
	```bash
	cd $TRT_OSSPATH
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
	make -j$(nproc)
	```

    **Example: Linux (aarch64) build with default cuda-12.1**
	```bash
	cd $TRT_OSSPATH
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain
	make -j$(nproc)
	```


## DEEPSTREAM 7.0
1. #### Download TensorRT OSS - DeepStream 7.0
	```bash
    cd TensorRTPlugin
	git clone -b release/8.6_efficientnmsx https://github.com/levipereira/TensorRT TensorRT
	cd TensorRT
	git submodule update --init --recursive
	```

2. #### Generate the TensorRT-OSS build container. 
    The TensorRT-OSS build container can be generated using the supplied Dockerfiles and build scripts. The build containers are configured for building TensorRT OSS out-of-the-box.

    **Ubuntu 20.04 on x86-64 with cuda-12.1 (DeepStream 7.0)** 
    ```bash
    ./docker/build.sh --file docker/ubuntu-20.04.Dockerfile --tag tensorrt-ubuntu20.04-cuda12.1
    ```
    **Example: Ubuntu 20.04 cross-compile for Jetson (aarch64) with cuda-11.4.2 (JetPack SDK)**
    ```bash
    ./docker/build.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag tensorrt-jetpack-cuda11.4
    ```
    **Example: Ubuntu 20.04 on aarch64 with cuda-11.8**
    ```bash
    ./docker/build.sh --file docker/ubuntu-20.04-aarch64.Dockerfile --tag tensorrt-aarch64-ubuntu20.04-cuda11.8 --cuda 11.8.0
    ```

3. #### Launch the TensorRT-OSS build container.
    **Example: Ubuntu 20.04 build container**
	```bash
	./docker/launch.sh --tag tensorrt-ubuntu20.04-cuda12.1 --gpus all
	```
	> NOTE:
  <br> `sudo` password for Ubuntu build containers is `nvidia`.
## Building TensorRT-OSS
* Generate Makefiles and build.

    **Example: Linux (x86-64) build with default cuda-12.1 - DeepStream 7.0**
	```bash
	cd $TRT_OSSPATH
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
	make -j$(nproc)
	```

    **Example: Linux (aarch64) build with default cuda-12.1**
	```bash
	cd $TRT_OSSPATH
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain
	make -j$(nproc)
	```
> NOTE:
* Required CMake build arguments are:
	- `TRT_LIB_DIR`: Path to the TensorRT installation directory containing libraries.
	- `TRT_OUT_DIR`: Output directory where generated build artifacts will be copied.
* Optional CMake build arguments:
	- `GPU_ARCHS`: GPU (SM) architectures to target. By default we generate CUDA code for all major SMs. Specific SM versions can be specified here as a quoted space-separated list to reduce compilation time and binary size. Table of compute capabilities of NVIDIA GPUs can be found [here](https://developer.nvidia.com/cuda-gpus). Examples:
        - NVidia A100: `-DGPU_ARCHS="80"`
        - Tesla T4, GeForce RTX 2080: `-DGPU_ARCHS="75"`
        - Titan V, Tesla V100: `-DGPU_ARCHS="70"`
        - Multiple SMs: `-DGPU_ARCHS="80 75"`

<br>

After building ends successfully, libnvinfer_plugin.so* will be generated under `${pwd}/out/`. <br>
exit from  docker container to host and copy `libnvinfer_plugin` 
```bash
## exit from container
exit
## copy lib to deepstream-yolov9/TensorRTPlugin
cp build/out/libnvinfer_plugin.so.8.6.1 ../../TensorRTPlugin/

## on this example we copied from 
# ~/deepstream-yolov9/TensorRTPlugin/TensorRT/build/out/libnvinfer_plugin.so.8.6.1  ~/deepstream-yolov9/TensorRTPlugin/

## optional to clean
## you can remove TensorRT dir
rm -rf deepstream-yolov9/TensorRTPlugin/TensorRT
```

 
 