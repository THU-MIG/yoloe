/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __SLICE_LAYER_H__
#define __SLICE_LAYER_H__

#include <string>

#include "NvInfer.h"

nvinfer1::ITensor* sliceLayer(int layerIdx, std::string& name, nvinfer1::ITensor* input, nvinfer1::Dims start,
    nvinfer1::Dims size, nvinfer1::Dims stride, nvinfer1::INetworkDefinition* network);

#endif
