from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ ==  '__main__':
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Use the model
    results = model.train(data="config.yaml", epochs=20)  # train the model