import fiftyone.zoo as foz
import os

if __name__ == "__main__":
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=300,
        shuffle=True,
    )

    os.makedirs("./coco_300_subset", exist_ok=True)
    dataset.export(export_dir="./coco_300_subset")