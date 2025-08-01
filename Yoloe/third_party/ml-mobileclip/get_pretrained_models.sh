#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
mkdir -p checkpoints
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_b.pt -P checkpoints
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt -P checkpoints
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt -P checkpoints
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt -P checkpoints
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt -P checkpoints
