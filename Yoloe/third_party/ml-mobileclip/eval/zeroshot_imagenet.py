#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""
Evaluation code is borrowed from https://github.com/mlfoundations/datacomp/blob/main/eval_utils/wds_eval.py
Licensed under MIT License, see ACKNOWLEDGEMENTS for details.
"""

import os
import argparse

import mobileclip
import torch
from clip_benchmark.datasets.builder import build_dataset
from clip_benchmark.metrics import zeroshot_classification as zsc


def parse_args(parser):
    parser.add_argument(
        "--model-arch",
        type=str,
        required=True,
        help="Specify model arch from the available choices.",
        choices=['mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2', 'mobileclip_b']
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Specify location of model checkpoint.",
    )
    return parser


def create_model(model_arch, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model_path = str(model_path)
    model, _, transform = mobileclip.create_model_and_transforms(
        model_arch, pretrained=model_path
    )
    model.eval()
    model = model.to(device)

    return model, transform, device


def create_webdataset(
    task, transform, data_root=None, dataset_len=None, batch_size=64, num_workers=4
):
    data_folder = f"wds_{task.replace('/','-')}_test"
    if data_root is None:
        data_root = f"https://hf-mirror.com/datasets/djghosh/{data_folder}/tree/main"
    else:
        data_root = os.path.join(data_root, data_folder)
    dataset = build_dataset(
        dataset_name=f"wds/{task}",
        root=data_root,
        transform=transform,
        split="test",
        download=False,
    )
    if dataset_len:
        dataset = dataset.with_length((dataset_len + batch_size - 1) // batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset.batched(batch_size),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
    )
    return dataset, dataloader


def evaluate_webdataset(
    task,
    model_arch,
    model_path,
    data_root=None,
    dataset_len=None,
    batch_size=64,
    num_workers=4,
):
    """Evaluate CLIP model on classification task."""

    # Create model
    model, transform, device = create_model(model_arch, model_path)

    # Load data
    dataset, dataloader = create_webdataset(
        task, transform, data_root, dataset_len, batch_size, num_workers
    )

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None
    assert (
        zeroshot_templates is not None and classnames is not None
    ), "Dataset does not support classification"

    # Evaluate
    classifier = zsc.zero_shot_classifier(
        model,
        mobileclip.get_tokenizer(model_arch),
        classnames,
        zeroshot_templates,
        device,
    )
    logits, target = zsc.run_classification(
        model, classifier, dataloader, device, amp=False
    )

    # Compute metrics
    acc1, acc5 = zsc.accuracy(logits, target, topk=(1, 5))
    metrics = {
        "acc1": acc1,
        "acc5": acc5,
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webdataset evaluation script.")
    parser = parse_args(parser)
    args = parser.parse_args()

    metric = evaluate_webdataset(
        task="imagenet1k", model_arch=args.model_arch, model_path=args.model_path
    )
    print(f"ImageNet1k Eval Metrics: {metric}")
