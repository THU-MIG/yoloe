# dGPU installation

To install the DeepStream on dGPU (x86 platform), without docker, we need to do some steps to prepare the computer.

**NOTE**: Disable Secure Boot in the BIOS.

1. Purge all NVIDIA driver, CUDA, etc ($CUDA_PATH = Path to the CUDA installation)

```
sudo nvidia-uninstall
sudo $CUDA_PATH/bin/cuda-uninstaller
sudo apt-get remove --purge '*libnv*'
sudo apt-get remove --purge '*cudnn*'
sudo apt-get remove --purge '*nvidia*'
sudo apt-get remove --purge '*cuda*'
sudo apt autoremove --purge && sudo apt autoclean && sudo apt clean
```

2. Install the essential packages
```
sudo apt-get update
sudo apt-get install gcc make git libtool autoconf autogen pkg-config cmake python3 python3-dev python3-pip
sudo apt-get install linux-headers-$(uname -r)
```

3. Reboot

```
sudo reboot
```

<details><summary>DeepStream 7.1</summary>

### 1. Dependencies

```
sudo apt-get install dkms
sudo apt-get install libssl3 libssl-dev libgles2-mesa-dev libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev libjsoncpp-dev protobuf-compiler
```

### 2. CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

### 3. GCC 12

```
sudo apt-get install gcc-12 g++-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12
sudo update-initramfs -u
```

### 4. NVIDIA Driver

<details><summary>TITAN, GeForce RTX / GTX series and RTX / Quadro series</summary><blockquote>

- Download

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/560.35.03/NVIDIA-Linux-x86_64-560.35.03.run
  ```

<blockquote><details><summary>Laptop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-560.35.03.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-560.35.03.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd
  ```

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

</details></blockquote>

<blockquote><details><summary>Desktop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-560.35.03.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-560.35.03.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

</details></blockquote>

</blockquote></details>

<details><summary>Data center / Tesla series</summary><blockquote>

  - Download

    ```
    wget https://us.download.nvidia.com/tesla/535.183.06/NVIDIA-Linux-x86_64-535.183.06.run
    ```

  * Run

    ```
    sudo sh NVIDIA-Linux-x86_64-535.183.06.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
    ```

</blockquote></details>

### 5. CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run
sudo sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

### 6. TensorRT

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get install libnvinfer-dev=10.3.0.26-1+cuda12.5 libnvinfer-dispatch-dev=10.3.0.26-1+cuda12.5 libnvinfer-dispatch10=10.3.0.26-1+cuda12.5 libnvinfer-headers-dev=10.3.0.26-1+cuda12.5 libnvinfer-headers-plugin-dev=10.3.0.26-1+cuda12.5 libnvinfer-lean-dev=10.3.0.26-1+cuda12.5 libnvinfer-lean10=10.3.0.26-1+cuda12.5 libnvinfer-plugin-dev=10.3.0.26-1+cuda12.5 libnvinfer-plugin10=10.3.0.26-1+cuda12.5 libnvinfer-vc-plugin-dev=10.3.0.26-1+cuda12.5 libnvinfer-vc-plugin10=10.3.0.26-1+cuda12.5 libnvinfer10=10.3.0.26-1+cuda12.5 libnvonnxparsers-dev=10.3.0.26-1+cuda12.5 libnvonnxparsers10=10.3.0.26-1+cuda12.5 tensorrt-dev=10.3.0.26-1+cuda12.5 libnvinfer-samples=10.3.0.26-1+cuda12.5 libnvinfer-bin=10.3.0.26-1+cuda12.5 libcudnn9-cuda-12=9.3.0.75-1 libcudnn9-dev-cuda-12=9.3.0.75-1
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn9* python3-libnvinfer* uff-converter-tf* onnx-graphsurgeon* graphsurgeon-tf* tensorrt*
```

### 7. DeepStream SDK

DeepStream 7.1 for Servers and Workstations

```
wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/7.1/files?redirect=true&path=deepstream-7.1_7.1.0-1_amd64.deb' -O deepstream-7.1_7.1.0-1_amd64.deb
sudo apt-get install ./deepstream-7.1_7.1.0-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
sudo ln -snf /usr/local/cuda-12.6 /usr/local/cuda
```

### 8. Reboot

```
sudo reboot
```

</details>

<details><summary>DeepStream 7.0</summary>

### 1. Dependencies

```
sudo apt-get install dkms
sudo apt-get install libssl3 libssl-dev libgles2-mesa-dev libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev libjsoncpp-dev protobuf-compiler
```

### 2. CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

### 3. GCC 12

```
sudo apt-get install gcc-12 g++-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12
sudo update-initramfs -u
```

### 4. NVIDIA Driver

<details><summary>TITAN, GeForce RTX / GTX series and RTX / Quadro series</summary><blockquote>

- Download

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.179/NVIDIA-Linux-x86_64-535.179.run
  ```

<blockquote><details><summary>Laptop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-535.179.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-535.179.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd
  ```

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

</details></blockquote>

<blockquote><details><summary>Desktop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-535.179.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-535.179.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

</details></blockquote>

</blockquote></details>

<details><summary>Data center / Tesla series</summary><blockquote>

  - Download

    ```
    wget https://us.download.nvidia.com/tesla/535.161.08/NVIDIA-Linux-x86_64-535.161.08.run
    ```

  * Run

    ```
    sudo sh NVIDIA-Linux-x86_64-535.161.08.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
    ```

</blockquote></details>

### 5. CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
sudo sh cuda_12.2.2_535.104.05_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

### 6. TensorRT

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get install --no-install-recommends libnvinfer-lean8=8.6.1.6-1+cuda12.0 libnvinfer-vc-plugin8=8.6.1.6-1+cuda12.0 libnvinfer-headers-dev=8.6.1.6-1+cuda12.0 libnvinfer-dev=8.6.1.6-1+cuda12.0 libnvinfer-headers-plugin-dev=8.6.1.6-1+cuda12.0 libnvinfer-plugin-dev=8.6.1.6-1+cuda12.0 libnvonnxparsers-dev=8.6.1.6-1+cuda12.0 libnvinfer-lean-dev=8.6.1.6-1+cuda12.0 libnvparsers-dev=8.6.1.6-1+cuda12.0 python3-libnvinfer-lean=8.6.1.6-1+cuda12.0 python3-libnvinfer-dispatch=8.6.1.6-1+cuda12.0 uff-converter-tf=8.6.1.6-1+cuda12.0 onnx-graphsurgeon=8.6.1.6-1+cuda12.0 libnvinfer-bin=8.6.1.6-1+cuda12.0 libnvinfer-dispatch-dev=8.6.1.6-1+cuda12.0 libnvinfer-dispatch8=8.6.1.6-1+cuda12.0 libnvonnxparsers-dev=8.6.1.6-1+cuda12.0 libnvonnxparsers8=8.6.1.6-1+cuda12.0 libnvinfer-vc-plugin-dev=8.6.1.6-1+cuda12.0 libnvinfer-samples=8.6.1.6-1+cuda12.0
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* python3-libnvinfer* uff-converter-tf* onnx-graphsurgeon*
```

### 7. DeepStream SDK

DeepStream 7.0 for Servers and Workstations

```
wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/7.0/files?redirect=true&path=deepstream-7.0_7.0.0-1_amd64.deb' -O deepstream-7.0_7.0.0-1_amd64.deb
sudo apt-get install ./deepstream-7.0_7.0.0-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
sudo ln -snf /usr/local/cuda-12.2 /usr/local/cuda
```

### 8. Reboot

```
sudo reboot
```

</details>

<details><summary>DeepStream 6.4</summary>

### 1. Dependencies

```
sudo apt-get install dkms
sudo apt-get install libssl3 libssl-dev libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev libjsoncpp-dev protobuf-compiler
```

### 2. CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

### 3. GCC 12

```
sudo apt-get install gcc-12 g++-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12
sudo update-initramfs -u
```

### 4. NVIDIA Driver

<details><summary>TITAN, GeForce RTX / GTX series and RTX / Quadro series</summary><blockquote>

- Download

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.179/NVIDIA-Linux-x86_64-535.179.run
  ```

<blockquote><details><summary>Laptop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-535.179.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-535.179.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd
  ```

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

</details></blockquote>

<blockquote><details><summary>Desktop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-535.179.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-535.179.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

</details></blockquote>

</blockquote></details>

<details><summary>Data center / Tesla series</summary><blockquote>

  - Download

    ```
    wget https://us.download.nvidia.com/tesla/535.104.12/NVIDIA-Linux-x86_64-535.104.12.run
    ```

  * Run

    ```
    sudo sh NVIDIA-Linux-x86_64-535.104.12.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
    ```

</blockquote></details>

### 5. CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
sudo sh cuda_12.2.2_535.104.05_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

### 6. TensorRT

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get install libnvinfer8=8.6.1.6-1+cuda12.0 libnvinfer-plugin8=8.6.1.6-1+cuda12.0 libnvparsers8=8.6.1.6-1+cuda12.0 libnvonnxparsers8=8.6.1.6-1+cuda12.0 libnvinfer-bin=8.6.1.6-1+cuda12.0 libnvinfer-dev=8.6.1.6-1+cuda12.0 libnvinfer-plugin-dev=8.6.1.6-1+cuda12.0 libnvparsers-dev=8.6.1.6-1+cuda12.0 libnvonnxparsers-dev=8.6.1.6-1+cuda12.0 libnvinfer-samples=8.6.1.6-1+cuda12.0 libcudnn8=8.9.4.25-1+cuda12.2 libcudnn8-dev=8.9.4.25-1+cuda12.2 libnvinfer-headers-dev=8.6.1.6-1+cuda12.0 libnvinfer-lean-dev=8.6.1.6-1+cuda12.0 libnvinfer-headers-plugin-dev=8.6.1.6-1+cuda12.0 libnvinfer-dispatch-dev=8.6.1.6-1+cuda12.0 libnvinfer-vc-plugin-dev=8.6.1.6-1+cuda12.0
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* python3-libnvinfer*
```

### 7. DeepStream SDK

DeepStream 7.0 for Servers and Workstations

```
wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/6.4/files?redirect=true&path=deepstream-6.4_6.4.0-1_amd64.deb' -O deepstream-6.4_6.4.0-1_amd64.deb
sudo apt-get install ./deepstream-6.4_6.4.0-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
sudo ln -snf /usr/local/cuda-12.2 /usr/local/cuda
```

### 8. Reboot

```
sudo reboot
```

</details>

<details><summary>DeepStream 6.3</summary>

### 1. Dependencies

```
sudo apt-get install dkms
sudo apt-get install libssl1.1 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev libjsoncpp-dev protobuf-compiler
```

### 2. CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

### 3. NVIDIA Driver

<details><summary>TITAN, GeForce RTX / GTX series and RTX / Quadro series</summary><blockquote>

- Download

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.125.06/NVIDIA-Linux-x86_64-525.125.06.run
  ```

<blockquote><details><summary>Laptop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-525.125.06.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-525.125.06.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd
  ```

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

</details></blockquote>

<blockquote><details><summary>Desktop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-525.125.06.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-525.125.06.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

</details></blockquote>

</blockquote></details>

<details><summary>Data center / Tesla series</summary><blockquote>

  - Download

    ```
    wget https://us.download.nvidia.com/tesla/525.125.06/NVIDIA-Linux-x86_64-525.125.06.run
    ```

  * Run

    ```
    sudo sh NVIDIA-Linux-x86_64-525.125.06.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
    ```

</blockquote></details>

### 4. CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

### 5. TensorRT

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install libnvinfer8=8.5.3-1+cuda11.8 libnvinfer-plugin8=8.5.3-1+cuda11.8 libnvparsers8=8.5.3-1+cuda11.8 libnvonnxparsers8=8.5.3-1+cuda11.8 libnvinfer-bin=8.5.3-1+cuda11.8 libnvinfer-dev=8.5.3-1+cuda11.8 libnvinfer-plugin-dev=8.5.3-1+cuda11.8 libnvparsers-dev=8.5.3-1+cuda11.8 libnvonnxparsers-dev=8.5.3-1+cuda11.8 libnvinfer-samples=8.5.3-1+cuda11.8 libcudnn8=8.7.0.84-1+cuda11.8 libcudnn8-dev=8.7.0.84-1+cuda11.8 python3-libnvinfer=8.5.3-1+cuda11.8 python3-libnvinfer-dev=8.5.3-1+cuda11.8
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* python3-libnvinfer*
```

### 6. DeepStream SDK

DeepStream 6.3 for Servers and Workstations

```
wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/6.3/files?redirect=true&path=deepstream-6.3_6.3.0-1_amd64.deb' -O deepstream-6.3_6.3.0-1_amd64.deb
sudo apt-get install ./deepstream-6.3_6.3.0-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
sudo ln -snf /usr/local/cuda-12.1 /usr/local/cuda
```

### 7. Reboot

```
sudo reboot
```

</details>

<details><summary>DeepStream 6.2</summary>

### 1. Dependencies

```
sudo apt-get install dkms
sudo apt-get install libssl1.1 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev libjsoncpp-dev protobuf-compiler
```

### 2. CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

### 3. NVIDIA Driver

<details><summary>TITAN, GeForce RTX / GTX series and RTX / Quadro series</summary><blockquote>

- Download

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.147.05/NVIDIA-Linux-x86_64-525.147.05.run
  ```

<blockquote><details><summary>Laptop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-525.147.05.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-525.147.05.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd
  ```

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

</details></blockquote>

<blockquote><details><summary>Desktop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-525.147.05.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-525.147.05.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

</details></blockquote>

</blockquote></details>

<details><summary>Data center / Tesla series</summary><blockquote>

  - Download

    ```
    wget https://us.download.nvidia.com/tesla/525.85.12/NVIDIA-Linux-x86_64-525.85.12.run
    ```

  * Run

    ```
    sudo sh NVIDIA-Linux-x86_64-525.85.12.run --no-cc-version-check --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
    ```

</blockquote></details>

### 4. CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

### 5. TensorRT

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install libnvinfer8=8.5.2-1+cuda11.8 libnvinfer-plugin8=8.5.2-1+cuda11.8 libnvparsers8=8.5.2-1+cuda11.8 libnvonnxparsers8=8.5.2-1+cuda11.8 libnvinfer-bin=8.5.2-1+cuda11.8 libnvinfer-dev=8.5.2-1+cuda11.8 libnvinfer-plugin-dev=8.5.2-1+cuda11.8 libnvparsers-dev=8.5.2-1+cuda11.8 libnvonnxparsers-dev=8.5.2-1+cuda11.8 libnvinfer-samples=8.5.2-1+cuda11.8 libcudnn8=8.7.0.84-1+cuda11.8 libcudnn8-dev=8.7.0.84-1+cuda11.8 python3-libnvinfer=8.5.2-1+cuda11.8 python3-libnvinfer-dev=8.5.2-1+cuda11.8
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* python3-libnvinfer*
```

### 6. DeepStream SDK

Download from the [NVIDIA website](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived): DeepStream 6.2 for Servers and Workstations (.deb)

```
sudo apt-get install ./deepstream-6.2_6.2.0-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
sudo ln -snf /usr/local/cuda-11.8 /usr/local/cuda
```

### 7. Reboot

```
sudo reboot
```

</details>

<details><summary>DeepStream 6.1.1</summary>

### 1. Dependencies

```
sudo apt-get install dkms
sudo apt-get install libssl1.1 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev
```

### 2. CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

### 3. NVIDIA Driver

<details><summary>TITAN, GeForce RTX / GTX series and RTX / Quadro series</summary><blockquote>

- Download

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/515.65.01/NVIDIA-Linux-x86_64-515.65.01.run
  ```

<blockquote><details><summary>Laptop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-515.65.01.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-515.65.01.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

</details></blockquote>

<blockquote><details><summary>Desktop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-515.65.01.run --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-515.65.01.run --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

</details></blockquote>

</blockquote></details>

<details><summary>Data center / Tesla series</summary><blockquote>

  - Download

    ```
    wget https://us.download.nvidia.com/tesla/515.65.01/NVIDIA-Linux-x86_64-515.65.01.run
    ```

  * Run

    ```
    sudo sh NVIDIA-Linux-x86_64-515.65.01.run --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
    ```

</blockquote></details>

### 4. CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
sudo sh cuda_11.7.1_515.65.01_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

### 5. TensorRT

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install libnvinfer8=8.4.1-1+cuda11.6 libnvinfer-plugin8=8.4.1-1+cuda11.6 libnvparsers8=8.4.1-1+cuda11.6 libnvonnxparsers8=8.4.1-1+cuda11.6 libnvinfer-bin=8.4.1-1+cuda11.6 libnvinfer-dev=8.4.1-1+cuda11.6 libnvinfer-plugin-dev=8.4.1-1+cuda11.6 libnvparsers-dev=8.4.1-1+cuda11.6 libnvonnxparsers-dev=8.4.1-1+cuda11.6 libnvinfer-samples=8.4.1-1+cuda11.6 libcudnn8=8.4.1.50-1+cuda11.6 libcudnn8-dev=8.4.1.50-1+cuda11.6 python3-libnvinfer=8.4.1-1+cuda11.6 python3-libnvinfer-dev=8.4.1-1+cuda11.6
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* python3-libnvinfer*
```

### 6. DeepStream SDK

Download from the [NVIDIA website](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived): DeepStream 6.1.1 for Servers and Workstations (.deb)

```
sudo apt-get install ./deepstream-6.1_6.1.1-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
sudo ln -snf /usr/local/cuda-11.7 /usr/local/cuda
```

### 7. Reboot

```
sudo reboot
```

</details>

<details><summary>DeepStream 6.1</summary>

### 1. Dependencies

```
sudo apt-get install dkms
sudo apt-get install libssl1.1 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev
```

### 2. CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

### 3. NVIDIA Driver

<details><summary>TITAN, GeForce RTX / GTX series and RTX / Quadro series</summary><blockquote>

- Download

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/510.47.03/NVIDIA-Linux-x86_64-510.47.03.run
  ```

<blockquote><details><summary>Laptop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-510.47.03.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-510.47.03.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

</details></blockquote>

<blockquote><details><summary>Desktop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-510.47.03.run --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-510.47.03.run --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

</details></blockquote>

</blockquote></details>

<details><summary>Data center / Tesla series</summary><blockquote>

  - Download

    ```
    wget https://us.download.nvidia.com/tesla/510.47.03/NVIDIA-Linux-x86_64-510.47.03.run
    ```

  * Run

    ```
    sudo sh NVIDIA-Linux-x86_64-510.47.03.run --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
    ```

</blockquote></details>

### 4. CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda_11.6.1_510.47.03_linux.run
sudo sh cuda_11.6.1_510.47.03_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

### 5. TensorRT

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install libnvinfer8=8.2.5-1+cuda11.4 libnvinfer-plugin8=8.2.5-1+cuda11.4 libnvparsers8=8.2.5-1+cuda11.4 libnvonnxparsers8=8.2.5-1+cuda11.4 libnvinfer-dev=8.2.5-1+cuda11.4 libnvinfer-plugin-dev=8.2.5-1+cuda11.4 libnvparsers-dev=8.2.5-1+cuda11.4 libnvonnxparsers-dev=8.2.5-1+cuda11.4 libcudnn8=8.4.0.27-1+cuda11.6 libcudnn8-dev=8.4.0.27-1+cuda11.6 python3-libnvinfer=8.2.5-1+cuda11.4 python3-libnvinfer-dev=8.2.5-1+cuda11.4
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* python3-libnvinfer*
```

### 6. DeepStream SDK

Download from the [NVIDIA website](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived): DeepStream 6.1 for Servers and Workstations (.deb)

```
sudo apt-get install ./deepstream-6.1_6.1.0-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
sudo ln -snf /usr/local/cuda-11.6 /usr/local/cuda
```

### 7. Reboot

```
sudo reboot
```

</details>

<details><summary>DeepStream 6.0.1 / 6.0</summary>

<details><summary>If you are using a laptop with newer Intel/AMD processors and your Graphics in Settings->Details->About tab is llvmpipe, please update the kernel.</summary>

```
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.11/amd64/linux-headers-5.11.0-051100_5.11.0-051100.202102142330_all.deb
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.11/amd64/linux-headers-5.11.0-051100-generic_5.11.0-051100.202102142330_amd64.deb
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.11/amd64/linux-image-unsigned-5.11.0-051100-generic_5.11.0-051100.202102142330_amd64.deb
wget https://kernel.ubuntu.com/~kernel-ppa/mainline/v5.11/amd64/linux-modules-5.11.0-051100-generic_5.11.0-051100.202102142330_amd64.deb
sudo dpkg -i  *.deb
sudo reboot
```

</details>

### 1. Dependencies

```
sudo apt-get install libssl1.0.0 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4
```

**NOTE**: Install DKMS (only if you are using the default Ubuntu kernel)

```
sudo apt-get install dkms
```

### 2. CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

### 3. NVIDIA Driver

<details><summary>TITAN, GeForce RTX / GTX series and RTX / Quadro series</summary><blockquote>

- Download

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/470.129.06/NVIDIA-Linux-x86_64-470.129.06.run
  ```

<blockquote><details><summary>Laptop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-470.129.06.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

  **NOTE**: Remove --dkms flag if you installed the 5.11.0 kernel.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-470.129.06.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: Remove --dkms flag if you installed the 5.11.0 kernel.

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

</details></blockquote>

<blockquote><details><summary>Desktop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-470.129.06.run --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

  **NOTE**: This step will disable the nouveau drivers.

  **NOTE**: Remove --dkms flag if you installed the 5.11.0 kernel.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-470.129.06.run --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

  **NOTE**: Remove --dkms flag if you installed the 5.11.0 kernel.

</details></blockquote>

</blockquote></details>

<details><summary>Data center / Tesla series</summary><blockquote>

  - Download

    ```
    wget https://us.download.nvidia.com/tesla/470.129.06/NVIDIA-Linux-x86_64-470.129.06.run
    ```

  * Run

    ```
    sudo sh NVIDIA-Linux-x86_64-470.129.06.run --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
    ```

    **NOTE**: Remove --dkms flag if you installed the 5.11.0 kernel.

</blockquote></details>

### 4. CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda_11.4.1_470.57.02_linux.run
sudo sh cuda_11.4.1_470.57.02_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

### 5. TensorRT

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get install libnvinfer8=8.0.1-1+cuda11.3 libnvinfer-plugin8=8.0.1-1+cuda11.3 libnvparsers8=8.0.1-1+cuda11.3 libnvonnxparsers8=8.0.1-1+cuda11.3 libnvinfer-dev=8.0.1-1+cuda11.3 libnvinfer-plugin-dev=8.0.1-1+cuda11.3 libnvparsers-dev=8.0.1-1+cuda11.3 libnvonnxparsers-dev=8.0.1-1+cuda11.3 libcudnn8=8.2.1.32-1+cuda11.3 libcudnn8-dev=8.2.1.32-1+cuda11.3 python3-libnvinfer=8.0.1-1+cuda11.3 python3-libnvinfer-dev=8.0.1-1+cuda11.3
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* python3-libnvinfer*
```

### 6. DeepStream SDK

Download from the [NVIDIA website](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived):

* DeepStream 6.0.1 for Servers and Workstations (.deb)

  ```
  sudo apt-get install ./deepstream-6.0_6.0.1-1_amd64.deb
  ```

* DeepStream 6.0 for Servers and Workstations (.deb)

  ```
  sudo apt-get install ./deepstream-6.0_6.0.0-1_amd64.deb
  ```

* Run

  ```
  rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
  sudo ln -snf /usr/local/cuda-11.4 /usr/local/cuda
  ```

### 7. Reboot

```
sudo reboot
```

</details>

<details><summary>DeepStream 5.1</summary>

### 1. Dependencies

```
sudo apt-get install dkms
sudo apt-get install libssl1.0.0 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4=2.11-1
```

### 2. CUDA Keyring

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```

### 3. NVIDIA Driver

<details><summary>TITAN, GeForce RTX / GTX series and RTX / Quadro series</summary><blockquote>

- Download

  ```
  wget https://us.download.nvidia.com/XFree86/Linux-x86_64/460.32.03/NVIDIA-Linux-x86_64-460.32.03.run
  ```

<blockquote><details><summary>Laptop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-460.32.03.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-460.32.03.run --silent --disable-nouveau --dkms --install-libglvnd
  ```

**NOTE**: If you are using a laptop with NVIDIA Optimius, run

```
sudo apt-get install nvidia-prime
sudo prime-select nvidia
```

</details></blockquote>

<blockquote><details><summary>Desktop</summary>

* Run

  ```
  sudo sh NVIDIA-Linux-x86_64-460.32.03.run --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

  **NOTE**: This step will disable the nouveau drivers.

* Reboot

  ```
  sudo reboot
  ```

* Install

  ```
  sudo sh NVIDIA-Linux-x86_64-460.32.03.run --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
  ```

</details></blockquote>

</blockquote></details>

<details><summary>Data center / Tesla series</summary><blockquote>

  - Download

    ```
    wget https://us.download.nvidia.com/tesla/460.32.03/NVIDIA-Linux-x86_64-460.32.03.run
    ```

  * Run

    ```
    sudo sh NVIDIA-Linux-x86_64-460.32.03.run --silent --disable-nouveau --dkms --install-libglvnd --run-nvidia-xconfig
    ```

</blockquote></details>

### 4. CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sudo sh cuda_11.1.1_455.32.00_linux.run --silent --toolkit
```

* Export environment variables

  ```
  echo $'export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc && source ~/.bashrc
  ```

### 5. TensorRT

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get install libnvinfer7=7.2.2-1+cuda11.1 libnvinfer-plugin7=7.2.2-1+cuda11.1 libnvparsers7=7.2.2-1+cuda11.1 libnvonnxparsers7=7.2.2-1+cuda11.1 libnvinfer-dev=7.2.2-1+cuda11.1 libnvinfer-plugin-dev=7.2.2-1+cuda11.1 libnvparsers-dev=7.2.2-1+cuda11.1 libnvonnxparsers-dev=7.2.2-1+cuda11.1 libcudnn8=8.0.5.39-1+cuda11.1 libcudnn8-dev=8.0.5.39-1+cuda11.1 python3-libnvinfer=7.2.2-1+cuda11.1 python3-libnvinfer-dev=7.2.2-1+cuda11.1
sudo apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* python3-libnvinfer*
```

### 6. DeepStream SDK

Download from the [NVIDIA website](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived): DeepStream 5.1 for Servers and Workstations (.deb)

```
sudo apt-get install ./deepstream-5.1_5.1.0-1_amd64.deb
rm ${HOME}/.cache/gstreamer-1.0/registry.x86_64.bin
sudo ln -snf /usr/local/cuda-11.1 /usr/local/cuda
```

### 7. Reboot

```
sudo reboot
```

</details>
