from ultralytics import YOLO

if __name__ == '__main__':
    # modelpath = r'D:\Yolov8\yolov8-detect-pt\yolov8s.pt'

    model = YOLO(r'C:\Users\32144\Desktop\ultralytics-8.2.0\ultralytics\cfg\models\v8\yolov8n.yaml')  # load a pretrained model (recommended for training)
 #   model.load('yolov8n.pt')
    # Train the model
    model.train(data=r'C:\Users\32144\Desktop\ultralytics-8.2.0\datasets\PCB_DATASET\datapcb.yaml',epochs=300, time=None, patience=0, batch=16, imgsz=640,)