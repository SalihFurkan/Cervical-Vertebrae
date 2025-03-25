from ultralytics import YOLO
import os
import shutil

# Load the trained model
model = YOLO("runs/detect/cvm_yolov88/weights/best.pt")

# Validate
#metrics = model.val()
#print(metrics.box.map)  # mAP@0.5:0.95

# Inference
dataset_path = "../Dataset/New/ALL"
predict_path = "../Dataset/New/ALL Predict"

for coll in os.listdir(dataset_path):

    if not coll.startswith("."):
    
        coll_path           = os.path.join(dataset_path, coll)
        predict_col_path    = os.path.join(predict_path, coll)
        
        for img_name in os.listdir(coll_path):
        
            if not img_name.startswith("."):
                
                img_path = os.path.join(coll_path, img_name)
#                model.predict(img_path, imgsz=1024, project=predict_col_path, save=True, save_crop=True, device='mps')
                

# Copy-Paste the cropped image to destination folder
save_path = "../Dataset/New/ALL RoI"

for coll in os.listdir(predict_path):

    if not coll.startswith("."):
        
        coll_path       = os.path.join(predict_path, coll)
        save_coll_path  = os.path.join(save_path, coll)
        
        for prd in os.listdir(coll_path):
        
            if not prd.startswith("."):
        
                img_folder_path = os.path.join(coll_path, prd, "crops", "cv")
                
                for img_name in os.listdir(img_folder_path):
                    
                    if not img_name.startswith("."):
                    
                        img_path        = os.path.join(img_folder_path, img_name)
                        save_img_path   = os.path.join(save_coll_path, img_name)
                        
                        shutil.copy(img_path, save_img_path)
            
            
