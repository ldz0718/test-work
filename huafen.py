import os
import random
import shutil
from tqdm import tqdm

def find_files(directory, extensions):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    return files

def split_dataset(image_dir, label_dir, output_dir, split_ratios):
    # 设置随机数种子，以确保每次划分数据集时结果相同
    random.seed(123)

    # 获取所有图片和标签文件的路径，存放到列表中
    image_extensions = [".jpg", ".jpeg", ".png"]
    label_extensions = [".txt", ".xml"]  # Adjust this if necessary

    image_files = find_files(image_dir, image_extensions)
    label_files = find_files(label_dir, label_extensions)

    # 将数据集文件列表分成训练集、验证集和测试集
    num_samples = len(image_files)
    num_train = int(num_samples * split_ratios["train"])
    num_val = int(num_samples * split_ratios["val"])
    num_test = num_samples - num_train - num_val

    random.shuffle(image_files)  # 打乱顺序，以确保随机性
    random.shuffle(label_files)  # 打乱顺序，以确保随机性

    train_images = image_files[:num_train]
    val_images = image_files[num_train:num_train + num_val]
    test_images = image_files[num_train + num_val:]

    train_labels = [img.replace(image_dir, label_dir).replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt").replace(".xml", ".txt") for img in train_images]
    val_labels = [img.replace(image_dir, label_dir).replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt").replace(".xml", ".txt") for img in val_images]
    test_labels = [img.replace(image_dir, label_dir).replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt").replace(".xml", ".txt") for img in test_images]

    # 创建输出目录和子目录
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split_name), exist_ok=True)

    # 复制数据到划分后的目录
    for images, labels, split_name in [(train_images, train_labels, "train"), (val_images, val_labels, "val"), (test_images, test_labels, "test")]:
        for image_file, label_file in tqdm(zip(images, labels), desc=f"Copying {split_name} data"):
            shutil.copy(image_file, os.path.join(output_dir, "images", split_name))
            shutil.copy(label_file, os.path.join(output_dir, "labels", split_name))

    print("Dataset split completed!")

def main():
    image_dir = r"C:\Users\32144\Desktop\ultralytics-8.2.0\NEU-DET\images"   #图片路径
    label_dir = r"C:\Users\32144\Desktop\ultralytics-8.2.0\NEU-DET\labels"	#标签路径
    output_dir = r"C:\Users\32144\Desktop\ultralytics-8.2.0\NEU-DET"  	#输出路径，没有创建会自行创建
    split_ratios = {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    }
    split_dataset(image_dir, label_dir, output_dir, split_ratios)

if __name__ == "__main__":
    main()


