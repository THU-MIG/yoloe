/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __SHORTCUT_LAYER_H__
#define __SHORTCUT_LAYER_H__

#include <map>

#include "NvInfer.h"

#include "slice_layer.h"
#include "activation_layer.h"

nvinfer1::ITensor* shortcutLayer(int layerIdx, std::string activation, std::string inputVol, std::string shortcutVol,
    std::map<std::string, std::string>& block, nvinfer1::ITensor* input, nvinfer1::ITensor* shortcut,
    nvinfer1::INetworkDefinition* network);

#endif
