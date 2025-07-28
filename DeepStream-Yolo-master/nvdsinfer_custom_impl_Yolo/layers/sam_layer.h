/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __SAM_LAYER_H__
#define __SAM_LAYER_H__

#include <map>

#include "NvInfer.h"

#include "activation_layer.h"

nvinfer1::ITensor* samLayer(int layerIdx, std::string activation, std::map<std::string, std::string>& block,
    nvinfer1::ITensor* input, nvinfer1::ITensor* samInput, nvinfer1::INetworkDefinition* network);

#endif
