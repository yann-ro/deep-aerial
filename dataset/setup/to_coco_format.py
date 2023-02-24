import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

from PIL import Image
from tqdm import tqdm


def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, "r") as f:
        labels_str = f.read().split()

    labels_ids = list(range(1, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))


class PascalVocDataset:
    def __init__(self) -> None:
        self.annotation_extension = "xml"

    def get_image_info(self, annotation_root, extract_num_from_imgid=True):
        path = annotation_root.findtext("path")
        if path is None:
            filename = annotation_root.findtext("filename")
        else:
            filename = os.path.basename(path)
            filename = filename.split("\\")[-1]
        img_name = os.path.basename(filename)
        img_id = os.path.splitext(img_name)[0]
        if extract_num_from_imgid and isinstance(img_id, str):
            img_id = int(re.findall(r"\d+", img_id)[0])

        size = annotation_root.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))

        image_info = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": img_id,
        }
        return image_info

    def get_coco_annotation(self, obj, label2id):
        label = obj.findtext("name")
        assert label in label2id, f"Error: {label} is not in label2id !"
        category_id = label2id[label]

        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.findtext("xmin"))) - 1
        ymin = int(float(bndbox.findtext("ymin"))) - 1
        xmax = int(float(bndbox.findtext("xmax")))
        ymax = int(float(bndbox.findtext("ymax")))
        assert (
            xmax > xmin and ymax > ymin
        ), f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
        o_width = xmax - xmin
        o_height = ymax - ymin

        ann = {
            "area": o_width * o_height,
            "iscrowd": 0,
            "bbox": [xmin, ymin, o_width, o_height],
            "category_id": category_id,
            "ignore": 0,
            "segmentation": [],
        }
        return ann

    def convert_to_cocojson(
        self,
        annotation_paths: List[str],
        label2id: Dict[str, int],
        output_jsonpath: str,
        extract_num_from_imgid: bool = True,
    ):
        output_json_dict = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": [],
        }
        bnd_id = 1

        annotation_paths = [path for path in annotation_paths if "xml" in path]

        for a_path in tqdm(annotation_paths):
            ann_tree = ET.parse(a_path)
            ann_root = ann_tree.getroot()

            img_info = self.get_image_info(
                annotation_root=ann_root, extract_num_from_imgid=extract_num_from_imgid
            )
            img_id = img_info["id"]
            output_json_dict["images"].append(img_info)

            for obj in ann_root.findall("object"):
                ann = self.get_coco_annotation(obj=obj, label2id=label2id)
                ann.update({"image_id": img_id, "id": bnd_id})
                output_json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1

        for label, label_id in label2id.items():
            category_info = {"supercategory": "none", "id": label_id, "name": label}
            output_json_dict["categories"].append(category_info)

        with open(output_jsonpath, "w") as f:
            output_json = json.dumps(output_json_dict)
            f.write(output_json)


class YoloDataset:
    def __init__(self) -> None:
        self.annotation_extension = "txt"

    def get_image_info(self, filename_label, img_root, img_id):
        filename = filename_label.replace(".txt", ".jpg")

        with Image.open(os.path.join(img_root, filename)) as im:
            width, height = im.size

        image_info = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": img_id,
        }
        return image_info

    def get_coco_annotation(self, labels, label2id, img_width, img_height):
        print(labels)
        label, xmin, ymin, width, height = labels

        assert label in label2id, f"Error: {label} is not in label2id !"
        category_id = label2id[label]

        width = int(float(width))
        height = int(float(height))
        xmin = int(float(xmin) * img_width - width / 2)
        ymin = int(float(ymin) * img_height - height / 2)

        ann = {
            "area": width * height,
            "iscrowd": 0,
            "bbox": [xmin, ymin, width, height],
            "category_id": category_id,
            "ignore": 0,
            "segmentation": [],
        }
        return ann

    def convert_to_cocojson(
        self,
        annotation_paths: List[str],
        label2id: Dict[str, int],
        output_jsonpath: str,
        extract_num_from_imgid: bool = True,
    ):
        output_json_dict = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": [],
        }
        bnd_id = 1

        annotation_paths = [path for path in annotation_paths if "txt" in path]

        for img_id, a_path in tqdm(enumerate(annotation_paths)):
            # images
            filename_label = os.path.basename(a_path)
            img_root = os.path.dirname(a_path).replace("labels", "images")
            img_info = self.get_image_info(filename_label, img_root, img_id)
            output_json_dict["images"].append(img_info)

            # annotations
            with open(a_path, "r") as f:
                labels_str = f.read().split()

            labels_str = [label_str.split(" ") for label_str in labels_str]

            for labels in labels_str:
                ann = self.get_coco_annotation(
                    labels, label2id, img_info["width"], img_info["height"]
                )
                ann.update({"image_id": img_id, "id": bnd_id})
                output_json_dict["annotations"].append(ann)

                bnd_id += 1

        # categories
        for label, label_id in label2id.items():
            category_info = {"supercategory": "none", "id": label_id, "name": label}
            output_json_dict["categories"].append(category_info)

        with open(output_jsonpath, "w") as f:
            output_json = json.dumps(output_json_dict)
            f.write(output_json)


def convert_to_coco(ann_dir, labels_path, input_type="xml", output_name="labels.json"):
    label2id = get_label2id(labels_path=labels_path)

    ann_paths = [os.path.join(ann_dir, file_name) for file_name in os.listdir(ann_dir)]

    output_jsonpath = os.path.join(os.path.dirname(ann_dir), output_name)

    if input_type == "xml":
        input_dataset = PascalVocDataset()
        input_dataset.convert_to_cocojson(
            annotation_paths=ann_paths,
            label2id=label2id,
            output_jsonpath=output_jsonpath,
        )
    if input_type == "txt":
        input_dataset = YoloDataset()
        input_dataset.convert_to_cocojson(
            annotation_paths=ann_paths,
            label2id=label2id,
            output_jsonpath=output_jsonpath,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", type=str, help="path")
    parser.add_argument("-l", "--label_name_path", type=str, help="label_path")
    parser.add_argument("-d", "--dataset_type", type=str, help="xml or txt")

    args = parser.parse_args()

    convert_to_coco(
        ann_dir=args.path,
        labels_path=args.label_name_path,
        input_type=args.dataset_type,
    )
