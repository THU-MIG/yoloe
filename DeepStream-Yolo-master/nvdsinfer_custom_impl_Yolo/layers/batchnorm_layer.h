/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __BATCHNORM_LAYER_H__
#define __BATCHNORM_LAYER_H__

#include <map>
#include <vector>

#include "NvInfer.h"

#include "activation_layer.h"

nvinfer1::ITensor* batchnormLayer(int layerIdx, std::map<std::string, std::string>& block, std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network);

#endif
