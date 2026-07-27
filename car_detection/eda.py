import os
import glob
import cv2
import matplotlib.pyplot as plt

def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.shape[1], image.shape[0]
    
    plotted_image = image.copy()
    
    for ann in annotations:
        if len(ann) == 5:
            class_id, x_center, y_center, width, height = ann
            x_center, width = x_center * w, width * w
            y_center, height = y_center * h, height * h
            
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
        else:
            class_id = ann[0]
            points = np.array(ann[1:]).reshape(-1, 2)
            points[:, 0] *= w
            points[:, 1] *= h
            
            x1, y1 = int(np.min(points[:, 0])), int(np.min(points[:, 1]))
            x2, y2 = int(np.max(points[:, 0])), int(np.max(points[:, 1]))
            
            # Optionally draw the polygon
            pts = points.astype(np.int32)
            cv2.polylines(plotted_image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            
        cv2.rectangle(plotted_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(plotted_image, "Plate", (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return plotted_image

import numpy as np

def visualize_dataset(dataset_path=r"dataset/dataset_indo/Indonesian License Plate Dataset", num_images=4):
    image_paths = glob.glob(os.path.join(dataset_path, "images", "train", "*.jpg"))
    if not image_paths:
        print("No images found!")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(min(num_images, len(image_paths))):
        img_path = image_paths[i]
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                annotations = [list(map(float, line.strip().split())) for line in f.readlines()]
            image = plot_bounding_box(image, annotations)
            
        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(os.path.basename(img_path))
        
    plt.tight_layout()
    plt.savefig("eda_output.png")
    print("EDA visualization saved to eda_output.png")

if __name__ == "__main__":
    visualize_dataset()
