/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __REORG_LAYER_H__
#define __REORG_LAYER_H__

#include <map>
#include <string>

#include "NvInfer.h"

#include "slice_layer.h"

nvinfer1::ITensor* reorgLayer(int layerIdx, std::map<std::string, std::string>& block, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network);

#endif
