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

#ifndef __UTILS_H__
#define __UTILS_H__

#include <map>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include <fstream>

#include "NvInfer.h"

std::string trim(std::string s);

float clamp(const float val, const float minVal, const float maxVal);

bool fileExists(const std::string fileName, bool verbose = true);

std::vector<float> loadWeights(const std::string weightsFilePath);

std::string dimsToString(const nvinfer1::Dims d);

int getNumChannels(nvinfer1::ITensor* t);

void printLayerInfo(
    std::string layerIndex, std::string layerName, std::string layerInput,  std::string layerOutput,
    std::string weightPtr);

#endif
