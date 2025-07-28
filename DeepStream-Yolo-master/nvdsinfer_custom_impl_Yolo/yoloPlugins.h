/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __YOLO_PLUGINS__
#define __YOLO_PLUGINS__

#include <cuda_runtime_api.h>

#include "yolo.h"

#define CUDA_CHECK(status) {                                                                                           \
  if (status != 0) {                                                                                                   \
    std::cout << "CUDA failure: " << cudaGetErrorString(status) << " in file " << __FILE__  << " at line "  <<         \
        __LINE__ << std::endl;                                                                                         \
    abort();                                                                                                           \
  }                                                                                                                    \
}

namespace {
  const char* YOLOLAYER_PLUGIN_VERSION {"1"};
  const char* YOLOLAYER_PLUGIN_NAME {"YoloLayer_TRT"};
} // namespace

class YoloLayer : public nvinfer1::IPluginV2DynamicExt {
  public:
    YoloLayer(const void* data, size_t length);

    YoloLayer(const uint& netWidth, const uint& netHeight, const uint& numClasses, const uint& newCoords,
        const std::vector<TensorInfo>& yoloTensors, const uint64_t& outputSize);

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    int initialize() noexcept override { return 0; }

    void terminate() noexcept override {}

    void destroy() noexcept override { delete this; }

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    int getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(INT index, const nvinfer1::DimsExprs* inputs, INT nbInputDims,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, INT nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, INT nbOutputs) const noexcept override { return 0; }

    bool supportsFormatCombination(INT pos, const nvinfer1::PluginTensorDesc* inOut, INT nbInputs, INT nbOutputs)
        noexcept override;

    const char* getPluginType() const noexcept override { return YOLOLAYER_PLUGIN_NAME; }

    const char* getPluginVersion() const noexcept override { return YOLOLAYER_PLUGIN_VERSION; }

    void setPluginNamespace(const char* pluginNamespace) noexcept override { m_Namespace = pluginNamespace; }

    const char* getPluginNamespace() const noexcept override { return m_Namespace.c_str(); }

    nvinfer1::DataType getOutputDataType(INT index, const nvinfer1::DataType* inputTypes, INT nbInputs) const noexcept
        override;
    
    void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
        nvinfer1::IGpuAllocator* gpuAllocator) noexcept override {}

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, INT nbInput,
        const nvinfer1::DynamicPluginTensorDesc* out, INT nbOutput) noexcept override;

    void detachFromContext() noexcept override {}
    
    INT enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc*  outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

  private:
    std::string m_Namespace {""};
    uint m_NetWidth {0};
    uint m_NetHeight {0};
    uint m_NumClasses {0};
    uint m_NewCoords {0};
    std::vector<TensorInfo> m_YoloTensors;
    uint64_t m_OutputSize {0};
};

class YoloLayerPluginCreator : public nvinfer1::IPluginCreator {
  public:
    YoloLayerPluginCreator() {}

    ~YoloLayerPluginCreator() {}

    const char* getPluginName() const noexcept override { return YOLOLAYER_PLUGIN_NAME; }

    const char* getPluginVersion() const noexcept override { return YOLOLAYER_PLUGIN_VERSION; }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
      std::cerr<< "YoloLayerPluginCreator::getFieldNames is not implemented" << std::endl;
      return nullptr;
    }

    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept
        override {
      std::cerr<< "YoloLayerPluginCreator::getFieldNames is not implemented";
      return nullptr;
    }

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength)
        noexcept override {
      std::cout << "Deserialize yoloLayer plugin: " << name << std::endl;
      return new YoloLayer(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) noexcept override { m_Namespace = libNamespace; }

    const char* getPluginNamespace() const noexcept override { return m_Namespace.c_str(); }

  private:
    std::string m_Namespace {""};
};

#endif // __YOLO_PLUGINS__
