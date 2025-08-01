################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
import platform
from threading import Lock
from cuda import cudart
from cuda import cuda

guard_platform_info = Lock()

import sys
import platform
from threading import Lock
from cuda import cudart
from cuda import cuda

guard_platform_info = Lock()

class PlatformInfo:
    def __init__(self):
        self.is_wsl_system = False
        self.wsl_verified = False
        self.is_integrated_gpu_system = False
        self.is_integrated_gpu_verified = False
        self.is_aarch64_platform = False
        self.is_aarch64_verified = False
        self.is_jetson_nano = False
        self.is_jetson = False

    def is_wsl(self):
        with guard_platform_info:
            if not self.wsl_verified:
                try:
                    with open("/proc/version", "r") as version_file:
                        version_info = version_file.readline().lower()
                        self.wsl_verified = True
                        if "microsoft" in version_info:
                            self.is_wsl_system = True
                except Exception as e:
                    print(f"ERROR: Opening /proc/version failed: {e}")
        return self.is_wsl_system
    
    def is_integrated_gpu(self):
        with guard_platform_info:
            if not self.is_integrated_gpu_verified:
                cuda_init_result, = cuda.cuInit(0)
                if cuda_init_result == cuda.CUresult.CUDA_SUCCESS:
                    device_count_result, num_devices = cuda.cuDeviceGetCount()
                    if device_count_result == cuda.CUresult.CUDA_SUCCESS:
                        if num_devices >= 1:
                            property_result, properties = cudart.cudaGetDeviceProperties(0)
                            if property_result == cuda.CUresult.CUDA_SUCCESS:
                                self.is_integrated_gpu_system = properties.integrated
                                self.is_integrated_gpu_verified = True
                            else:
                                print("ERROR: Getting cuda device property failed: {}".format(property_result))
                        else:
                            print("ERROR: No cuda devices found to check whether iGPU/dGPU")
                    else:
                        print("ERROR: Getting cuda device count failed: {}".format(device_count_result))
                else:
                    print("ERROR: Cuda init failed: {}".format(cuda_init_result))
        return self.is_integrated_gpu_system

    def is_platform_aarch64(self):
        if not self.is_aarch64_verified:
            if platform.uname()[4] == 'aarch64':
                self.is_aarch64_platform = True
            self.is_aarch64_verified = True
        return self.is_aarch64_platform

    def is_jetson_device(self):
        """Checks if the device is an NVIDIA Jetson."""
        if self.is_platform_aarch64():
            try:
                with open("/proc/device-tree/model", "r") as file:
                    model_info = file.read()
                    self.is_jetson = "NVIDIA Jetson" in model_info
            except FileNotFoundError:
                raise RuntimeError("ERROR: /proc/device-tree/model not found. "
                                   "Run Docker as privileged with the --privileged flag.")
        return self.is_jetson
    
    def is_jetson_nano_device(self):
        """Checks if the device is specifically a Jetson Nano."""
        if self.is_jetson_device():
            try:
                with open("/proc/device-tree/model", "r") as file:
                    model_info = file.read()
                    self.is_jetson_nano = "Orin Nano" in model_info
            except FileNotFoundError:
                raise RuntimeError("ERROR: /proc/device-tree/model not found. "
                                   "Run Docker as privileged with the --privileged flag.")
        return self.is_jetson_nano


sys.path.append('/opt/nvidia/deepstream/deepstream/lib')

