/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __CHANNELS_LAYER_H__
#define __CHANNELS_LAYER_H__

#include <map>
#include <string>

#include "NvInfer.h"

nvinfer1::ITensor* channelsLayer(int layerIdx, std::map<std::string, std::string>& block, nvinfer1::ITensor* input,
    nvinfer1::ITensor* implicitTensor, nvinfer1::INetworkDefinition* network);

#endif
