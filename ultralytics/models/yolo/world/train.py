# Ultralytics YOLO 🚀, AGPL-3.0 license

import itertools

from ultralytics.data import build_yolo_dataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import WorldModel, WorldSegModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import de_parallel
from copy import copy


class WorldTrainer(yolo.detect.DetectionTrainer):
    """
    A class to fine-tune a world model on a close-set dataset.

    Example:
        ```python
        from ultralytics.models.yolo.world import WorldModel

        args = dict(model="yolov8s-world.pt", data="coco8.yaml", epochs=3)
        trainer = WorldTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a WorldTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return WorldModel initialized with specified config and weights."""
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = WorldModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""
        batch = super().preprocess_batch(batch)
        batch["txt_feats"] = batch["texts"].to(self.device)
        return batch
