/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __ROUTE_LAYER_H__
#define __ROUTE_LAYER_H__

#include "../utils.h"

#include "slice_layer.h"

nvinfer1::ITensor* routeLayer(int layerIdx, std::string& layers, std::map<std::string, std::string>& block,
    std::vector<nvinfer1::ITensor*> tensorOutputs, nvinfer1::INetworkDefinition* network);

#endif
