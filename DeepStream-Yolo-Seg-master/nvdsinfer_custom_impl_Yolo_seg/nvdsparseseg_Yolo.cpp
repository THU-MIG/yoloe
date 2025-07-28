/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <cstring>

#include "nvdsinfer_custom_impl.h"

#include "utils.h"

#define NMS_THRESH 0.45;

extern "C" bool
NvDsInferParseYoloSeg(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList);

static void
addSegProposal(const float* masks, const uint& maskWidth, const uint& maskHeight, const uint& b,
    NvDsInferInstanceMaskInfo& obj)
{
  obj.mask = new float[maskHeight * maskWidth];
  obj.mask_width = maskWidth;
  obj.mask_height = maskHeight;
  obj.mask_size = sizeof(float) * maskHeight * maskWidth;

  const float* data = reinterpret_cast<const float*>(masks + b * maskHeight * maskWidth);
  memcpy(obj.mask, data, sizeof(float) * maskHeight * maskWidth);
}

static void
addBBoxProposal(const float& bx1, const float& by1, const float& bx2, const float& by2, const uint& netW, const uint& netH,
    const int& maxIndex, const float& maxProb, NvDsInferInstanceMaskInfo& obj)
{
  float x1 = clamp(bx1, 0, netW);
  float y1 = clamp(by1, 0, netH);
  float x2 = clamp(bx2, 0, netW);
  float y2 = clamp(by2, 0, netH);

  obj.left = x1;
  obj.width = clamp(x2 - x1, 0, netW);
  obj.top = y1;
  obj.height = clamp(y2 - y1, 0, netH);

  if (obj.width < 1 || obj.height < 1) {
      return;
  }

  obj.detectionConfidence = maxProb;
  obj.classId = maxIndex;
}

static std::vector<NvDsInferInstanceMaskInfo>
decodeTensorYoloSeg(const float* boxes, const float* scores, const float* classes, const float* masks,
    const uint& outputSize, const uint& maskWidth, const uint& maskHeight, const uint& netW, const uint& netH,
    const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferInstanceMaskInfo> objects;

  for (uint b = 0; b < outputSize; ++b) {
    float maxProb = scores[b];
    int maxIndex = (int) classes[b];

    if (maxProb < preclusterThreshold[maxIndex]) {
      continue;
    }

    float bx1 = boxes[b * 4 + 0];
    float by1 = boxes[b * 4 + 1];
    float bx2 = boxes[b * 4 + 2];
    float by2 = boxes[b * 4 + 3];

    NvDsInferInstanceMaskInfo obj;

    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, obj);
    addSegProposal(masks, maskWidth, maskHeight, b, obj);

    objects.push_back(obj);
  }

  return objects;
}

static bool
NvDsInferParseCustomYoloSeg(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& boxes = outputLayersInfo[0];
  const NvDsInferLayerInfo& scores = outputLayersInfo[1];
  const NvDsInferLayerInfo& classes = outputLayersInfo[2];
  const NvDsInferLayerInfo& masks = outputLayersInfo[3];

  const uint outputSize = boxes.inferDims.d[0];
  const uint maskWidth = masks.inferDims.d[2];
  const uint maskHeight = masks.inferDims.d[1];

  std::vector<NvDsInferInstanceMaskInfo> objects = decodeTensorYoloSeg((const float*) (boxes.buffer),
      (const float*) (scores.buffer), (const float*) (classes.buffer), (const float*) (masks.buffer), outputSize, maskWidth,
      maskHeight, networkInfo.width, networkInfo.height, detectionParams.perClassPreclusterThreshold);

  objectList = objects;

  return true;
}

extern "C" bool
NvDsInferParseYoloSeg(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  return NvDsInferParseCustomYoloSeg(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloSeg);
