import glob
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))


def run(func, this_iter, desc="Processing"):
    with ThreadPoolExecutor(max_workers=NUM_THREADS, thread_name_prefix='MyThread') as executor:
        results = list(
            tqdm(executor.map(func, this_iter), total=len(this_iter), desc=desc)
        )
    return results


# XML坐标格式转换成yolo坐标格式
def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def get_xml_classes(xml_path):
    f = open(xml_path)  # xml文件路径
    xml_text = f.read()
    root = ET.fromstring(xml_text)
    f.close()
    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls not in xml_classes:
            classes_file.write(cls + "\n")
            xml_classes.append(cls)


# 标记文件格式转换
def convert_xml2yolo(img_path):
    img_path = Path(img_path)

    xml_name = re.sub(r"\.(jpg|png|jpeg)$", ".xml", img_path.name)
    txt_name = re.sub(r"\.(jpg|png|jpeg)$", ".txt", img_path.name)
    xml_path = Path(xml_target_path) / xml_name
    txt_path = Path(save_path) / txt_name

    if xml_path.exists():
        out_file = open(txt_path, "w")  # 转换后的txt文件存放路径
        f = open(xml_path)  # xml文件路径
        xml_text = f.read()
        root = ET.fromstring(xml_text)
        f.close()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        if w == 0 or h == 0:
            # problem_xml.append(str(img_path.name))
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
            h, w, _ = img.shape
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in xml_classes:
                print(cls)
                continue
            cls_id = xml_classes.index(cls)
            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )
            try:
                bbox = convert((w, h), b)
            except:
                print(img_path)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bbox]) + "\n")
    else:
        print(f"{xml_path}不存在!")


if __name__ == "__main__":
    xml_target_path = r"C:\Users\32144\Desktop\ultralytics-8.2.0\NEU-DET\annotations"  # xml文件夹
    save_path = r"C:\Users\32144\Desktop\ultralytics-8.2.0\NEU-DET\labels"  # 转换后的txt文件存放文件夹
    images_path = r"C:\Users\32144\Desktop\ultralytics-8.2.0\NEU-DET\images"  # 图片文件夹
    classes_file = open(Path(xml_target_path).parents[0] / "classes.txt", "w")
    # -------------------------------------------- #
    # 第一步 获得xml所有种类
    # -------------------------------------------- #
    assert (Path(xml_target_path)).exists(), "Annotations文件夹不存在"
    xml_classes = []
    xml_list = glob.glob(os.path.join(xml_target_path, "*.[x][m][l]*"))
    run(get_xml_classes, xml_list)
    print(Path(xml_target_path).parents[0])
    print(xml_classes)
    # -------------------------------------------- #
    # 第二步 转换成YOLO txt
    # -------------------------------------------- #
    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True)
    file_list = glob.glob(os.path.join(images_path, "*.[jp][pn][gg]*"))
    run(convert_xml2yolo, file_list)

