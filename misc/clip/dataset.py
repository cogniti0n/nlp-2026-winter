from __future__ import annotations

import torch
import os
from typing import Tuple, Any, List

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions


class CocoCaptionsCLIP(CocoCaptions):
    def __getitem__(self, index):
        image, captions = super().__getitem__(index)
        caption = captions[0]
        return image, caption


def coco_collate_fn(batch):
    # batch: list[(image_tensor, caption_str)]
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(captions)


class ImageNetVal(Dataset):
    def __init__(
        self,
        val_dir: str,
        gt_path: str,
        transform=None,
        strict: bool = True,
        filename_prefix: str = "ILSVRC2012_val_",
        filename_ext: str = ".JPEG",
    ):
        self.val_dir = val_dir
        self.gt_path = gt_path
        self.transform = transform
        self.strict = strict
        self.filename_prefix = filename_prefix
        self.filename_ext = filename_ext

        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"val_dir not found: {val_dir}")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"gt_path not found: {gt_path}")

        # load labels
        labels: List[int] = []
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                y = int(s)
                if y < 1 or y > 1000:
                    raise ValueError(
                        f"Invalid label {y} in {gt_path} (expected 1..1000)"
                    )
                labels.append(y - 1)

        if strict and len(labels) != 50000:
            raise RuntimeError(
                f"Expected 50000 labels, got {len(labels)} from {gt_path}"
            )

        self.labels = labels

        n = len(self.labels)
        self.image_paths = []
        for i in range(1, n + 1):
            fname = f"{self.filename_prefix}{i:08d}{self.filename_ext}"
            fpath = os.path.join(val_dir, fname)
            if strict and not os.path.isfile(fpath):
                raise FileNotFoundError(f"Missing expected ImageNet val file: {fpath}")
            self.image_paths.append(fpath)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img_path = self.image_paths[idx]
        target = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class CC3M(Dataset):
    pass


import glob
import webdataset as wds


def get_cc3m_loader(args, preprocess_fn, num_workers, batch_size, world_size=1):
    shards = sorted(glob.glob(os.path.join(args.cc3m_path, "*.tar")))
    if not shards:
        raise FileNotFoundError

    print(f"# of shards: {len(shards)}")

    def decode_caption(t):
        if isinstance(t, bytes):
            return t.decode("utf-8", errors="ignore")
        if isinstance(t, str):
            return t
        if isinstance(t, dict):
            return t.get("caption", "") or t.get("text", "") or ""
        return ""

    dataset = (
        wds.WebDataset(shards, resampled=True, nodesplitter=wds.split_by_node)
        .shuffle(1000)
        .decode("pil", handler=wds.warn_and_continue)
        .rename(
            image="jpg;png", text="txt;caption;json"
        )  # Handle potential key variations
        .map_dict(image=preprocess_fn, text=decode_caption)
        .to_tuple("image", "text")
        .batched(batch_size, partial=False)
    )
    dataset = dataset.compose(wds.split_by_worker)

    epoch_length = 3_000_000
    num_batches = epoch_length // (batch_size * max(1, world_size))

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    ).with_length(num_batches)

    return loader


import webdataset as wds
import braceexpand


def get_datacomp_loader(
    args, preprocess_fn, num_workers, batch_size, world_size=1, epoch_length=12800000
):  # 10 epochs

    url = "https://huggingface.co/datasets/mlfoundations/datacomp_small/resolve/main/shards/{00000..01280}.tar"
    urls = list(braceexpand.braceexpand(url))

    dataset = (
        wds.WebDataset(urls, resampled=True, nodesplitter=wds.split_by_node)
        .shuffle(1000)
        .decode("pil", handler=wds.warn_and_continue)
        .to_tuple("jpg", "json")
        .map_tuple(preprocess_fn, lambda x: x["caption"])
        .with_epoch(
            epoch_length // max(1, world_size)
        )  # keep global epoch size across ranks
        .batched(batch_size, partial=False)
    )
    dataset = dataset.compose(wds.split_by_worker)

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    length = epoch_length // (batch_size * max(1, world_size))
    loader = loader.with_length(length)

    return loader


from torchvision.datasets import Flowers102, StanfordCars, Food101


def get_transfer_dataset(name: str, root: str, transform=None, split: str = "train"):
    if name == "flowers102":
        dataset = Flowers102(root=root, split=split, transform=transform, download=True)
    elif name == "stanfordcars":
        if split == "val":
            split = "test"
        dataset = StanfordCars(
            root=root, split=split, transform=transform, download=True
        )
    elif name == "food101":
        if split == "val":
            split = "test"
        dataset = Food101(root=root, split=split, transform=transform, download=True)
    else:
        raise ValueError(f"Unknown transfer dataset: {name}")
    return dataset


def get_classname_from_torchvision_dataset(ds):
    if (
        hasattr(ds, "classes")
        and isinstance(ds.classes, (list, tuple))
        and len(ds.classes) > 0
    ):
        return list(ds.classes)

    # some datasets use categories
    if (
        hasattr(ds, "categories")
        and isinstance(ds.categories, (list, tuple))
        and len(ds.categories) > 0
    ):
        return list(ds.categories)

    # last resort: create placeholder class names
    # (still allows training, but you lose semantic labels)
    n = len(set([ds[i][1] for i in range(min(len(ds), 2000))]))
    return [f"class_{i}" for i in range(n)]
