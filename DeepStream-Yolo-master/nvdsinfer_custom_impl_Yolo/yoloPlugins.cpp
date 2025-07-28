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

#include "yoloPlugins.h"

namespace {
  template <typename T>
  void write(char*& buffer, const T& val) {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
  }
  template <typename T>
  void read(const char*& buffer, T& val) {
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
  }
}

cudaError_t cudaYoloLayer_nc(const void* input, void* output, const uint& batchSize, const uint64_t& inputSize,
    const uint64_t& outputSize, const uint64_t& lastInputSize, const uint& netWidth, const uint& netHeight,
    const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses, const uint& numBBoxes,
    const float& scaleXY, const void* anchors, const void* mask, cudaStream_t stream);

cudaError_t cudaYoloLayer(const void* input, void* output, const uint& batchSize, const uint64_t& inputSize,
    const uint64_t& outputSize, const uint64_t& lastInputSize, const uint& netWidth, const uint& netHeight,
    const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses, const uint& numBBoxes,
    const float& scaleXY, const void* anchors, const void* mask, cudaStream_t stream);

cudaError_t cudaRegionLayer(const void* input, void* softmax, void* output, const uint& batchSize,
    const uint64_t& inputSize, const uint64_t& outputSize, const uint64_t& lastInputSize, const uint& netWidth,
    const uint& netHeight, const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses,
    const uint& numBBoxes, const void* anchors, cudaStream_t stream);

YoloLayer::YoloLayer(const void* data, size_t length) {
  const char* d = static_cast<const char*>(data);

  read(d, m_NetWidth);
  read(d, m_NetHeight);
  read(d, m_NumClasses);
  read(d, m_NewCoords);
  read(d, m_OutputSize);

  uint yoloTensorsSize;
  read(d, yoloTensorsSize);
  for (uint i = 0; i < yoloTensorsSize; ++i) {
    TensorInfo curYoloTensor;
    read(d, curYoloTensor.gridSizeX);
    read(d, curYoloTensor.gridSizeY);
    read(d, curYoloTensor.numBBoxes);
    read(d, curYoloTensor.scaleXY);

    uint anchorsSize;
    read(d, anchorsSize);
    for (uint j = 0; j < anchorsSize; ++j) {
      float result;
      read(d, result);
      curYoloTensor.anchors.push_back(result);
    }

    uint maskSize;
    read(d, maskSize);
    for (uint j = 0; j < maskSize; ++j) {
      int result;
      read(d, result);
      curYoloTensor.mask.push_back(result);
    }

    m_YoloTensors.push_back(curYoloTensor);
  }
};

YoloLayer::YoloLayer(const uint& netWidth, const uint& netHeight, const uint& numClasses, const uint& newCoords,
    const std::vector<TensorInfo>& yoloTensors, const uint64_t& outputSize) : m_NetWidth(netWidth),
    m_NetHeight(netHeight), m_NumClasses(numClasses), m_NewCoords(newCoords), m_YoloTensors(yoloTensors),
    m_OutputSize(outputSize)
{
  assert(m_NetWidth > 0);
  assert(m_NetHeight > 0);
  assert(m_NumClasses > 0);
  assert(m_OutputSize > 0);
};

nvinfer1::IPluginV2DynamicExt*
YoloLayer::clone() const noexcept
{
  return new YoloLayer(m_NetWidth, m_NetHeight, m_NumClasses, m_NewCoords, m_YoloTensors, m_OutputSize);
}

size_t
YoloLayer::getSerializationSize() const noexcept
{
  size_t totalSize = 0;

  totalSize += sizeof(m_NetWidth);
  totalSize += sizeof(m_NetHeight);
  totalSize += sizeof(m_NumClasses);
  totalSize += sizeof(m_NewCoords);
  totalSize += sizeof(m_OutputSize);

  uint yoloTensorsSize = m_YoloTensors.size();
  totalSize += sizeof(yoloTensorsSize);

  for (uint i = 0; i < yoloTensorsSize; ++i) {
    const TensorInfo& curYoloTensor = m_YoloTensors.at(i);
    totalSize += sizeof(curYoloTensor.gridSizeX);
    totalSize += sizeof(curYoloTensor.gridSizeY);
    totalSize += sizeof(curYoloTensor.numBBoxes);
    totalSize += sizeof(curYoloTensor.scaleXY);
    totalSize += sizeof(uint) + sizeof(curYoloTensor.anchors[0]) * curYoloTensor.anchors.size();
    totalSize += sizeof(uint) + sizeof(curYoloTensor.mask[0]) * curYoloTensor.mask.size();
  }

  return totalSize;
}

void
YoloLayer::serialize(void* buffer) const noexcept
{
  char* d = static_cast<char*>(buffer);

  write(d, m_NetWidth);
  write(d, m_NetHeight);
  write(d, m_NumClasses);
  write(d, m_NewCoords);
  write(d, m_OutputSize);

  uint yoloTensorsSize = m_YoloTensors.size();
  write(d, yoloTensorsSize);
  for (uint i = 0; i < yoloTensorsSize; ++i) {
    const TensorInfo& curYoloTensor = m_YoloTensors.at(i);
    write(d, curYoloTensor.gridSizeX);
    write(d, curYoloTensor.gridSizeY);
    write(d, curYoloTensor.numBBoxes);
    write(d, curYoloTensor.scaleXY);

    uint anchorsSize = curYoloTensor.anchors.size();
    write(d, anchorsSize);
    for (uint j = 0; j < anchorsSize; ++j) {
      write(d, curYoloTensor.anchors[j]);
    }

    uint maskSize = curYoloTensor.mask.size();
    write(d, maskSize);
    for (uint j = 0; j < maskSize; ++j) {
      write(d, curYoloTensor.mask[j]);
    }
  }
}

nvinfer1::DimsExprs
YoloLayer::getOutputDimensions(INT index, const nvinfer1::DimsExprs* inputs, INT nbInputDims,
    nvinfer1::IExprBuilder& exprBuilder)noexcept
{
  assert(index < 1);
  return nvinfer1::DimsExprs{3, {inputs->d[0], exprBuilder.constant(static_cast<int>(m_OutputSize)),
      exprBuilder.constant(6)}};
}

bool
YoloLayer::supportsFormatCombination(INT pos, const nvinfer1::PluginTensorDesc* inOut, INT nbInputs, INT nbOutputs)
    noexcept
{
  return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kFLOAT;
}

nvinfer1::DataType
YoloLayer::getOutputDataType(INT index, const nvinfer1::DataType* inputTypes, INT nbInputs) const noexcept
{
  assert(index < 1);
  return nvinfer1::DataType::kFLOAT;
}

void
YoloLayer::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, INT nbInput,
    const nvinfer1::DynamicPluginTensorDesc* out, INT nbOutput) noexcept
{
  assert(nbInput > 0);
  assert(in->desc.format == nvinfer1::PluginFormat::kLINEAR);
  assert(in->desc.dims.d != nullptr);
}

INT
YoloLayer::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc*  outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
  INT batchSize = inputDesc[0].dims.d[0];

  uint64_t lastInputSize = 0;

  uint yoloTensorsSize = m_YoloTensors.size();
  for (uint i = 0; i < yoloTensorsSize; ++i) {
    TensorInfo& curYoloTensor = m_YoloTensors.at(i);

    const uint numBBoxes = curYoloTensor.numBBoxes;
    const float scaleXY = curYoloTensor.scaleXY;
    const uint gridSizeX = curYoloTensor.gridSizeX;
    const uint gridSizeY = curYoloTensor.gridSizeY;
    const std::vector<float> anchors = curYoloTensor.anchors;
    const std::vector<int> mask = curYoloTensor.mask;

    void* d_anchors;
    void* d_mask;
    if (anchors.size() > 0) {
      CUDA_CHECK(cudaMalloc(&d_anchors, sizeof(float) * anchors.size()));
      CUDA_CHECK(cudaMemcpyAsync(d_anchors, anchors.data(), sizeof(float) * anchors.size(), cudaMemcpyHostToDevice,
          stream));
    }
    if (mask.size() > 0) {
      CUDA_CHECK(cudaMalloc(&d_mask, sizeof(int) * mask.size()));
      CUDA_CHECK(cudaMemcpyAsync(d_mask, mask.data(), sizeof(int) * mask.size(), cudaMemcpyHostToDevice, stream));
    }

    const uint64_t inputSize = (numBBoxes * (4 + 1 + m_NumClasses)) * gridSizeY * gridSizeX;

    if (mask.size() > 0) {
      if (m_NewCoords) {
        CUDA_CHECK(cudaYoloLayer_nc(inputs[i], outputs[0], batchSize, inputSize, m_OutputSize, lastInputSize,
            m_NetWidth, m_NetHeight, gridSizeX, gridSizeY, m_NumClasses, numBBoxes, scaleXY, d_anchors, d_mask,
            stream));
      }
      else {
        CUDA_CHECK(cudaYoloLayer(inputs[i], outputs[0], batchSize, inputSize, m_OutputSize, lastInputSize, m_NetWidth,
            m_NetHeight, gridSizeX, gridSizeY, m_NumClasses, numBBoxes, scaleXY, d_anchors, d_mask, stream));
      }
    }
    else {
      void* softmax;
      CUDA_CHECK(cudaMalloc(&softmax, sizeof(float) * inputSize * batchSize));
      CUDA_CHECK(cudaMemsetAsync((float*)softmax, 0, sizeof(float) * inputSize * batchSize, stream));

      CUDA_CHECK(cudaRegionLayer(inputs[i], softmax, outputs[0], batchSize, inputSize, m_OutputSize, lastInputSize,
          m_NetWidth, m_NetHeight, gridSizeX, gridSizeY, m_NumClasses, numBBoxes, d_anchors, stream));

      CUDA_CHECK(cudaFree(softmax));
    }

    if (anchors.size() > 0) {
      CUDA_CHECK(cudaFree(d_anchors));
    }
    if (mask.size() > 0) {
      CUDA_CHECK(cudaFree(d_mask));
    }

    lastInputSize += numBBoxes * gridSizeY * gridSizeX;
  }

  return 0;
}

REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);
