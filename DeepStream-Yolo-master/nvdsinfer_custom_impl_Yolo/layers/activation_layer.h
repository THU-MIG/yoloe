/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __ACTIVATION_LAYER_H__
#define __ACTIVATION_LAYER_H__

#include <string>

#include "NvInfer.h"

nvinfer1::ITensor* activationLayer(int layerIdx, std::string activation, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network, std::string layerName = "");

#endif
