from ultralytics import YOLO

if __name__ == '__main__':
    # modelpath = r'D:\Yolov8\yolov8-detect-pt\yolov8s.pt'

    model = YOLO('/root/autodl-tmp/yolov8/ultralytics/cfg/models/v8/yolov8n-GOLD.yaml')  # load a pretrained model (recommended for training)
 #   model.load('yolov8n.pt')
    # Train the model
    model.train(data='/root/autodl-tmp/yolov8/datasets/PCB_DATASET/data.yaml',epochs=300, time=None, patience=0, batch=16, imgsz=640,)

