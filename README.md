# ins_segm_cars


from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# Load YOLOv8 instance segmentation model
model_seg = YOLO("yolov8n-seg.pt")

# Your input image
image_path = "C:/Users/International/OneDrive/Рабочий стол/test_images_cars/image2.jpg"

# Run prediction
results_seg = model_seg(image_path)[0]

# Show segmentation results (masks + bounding boxes)
results_seg.show()

# Save output image to the same directory as input
input_dir = os.path.dirname(image_path)
output_path = os.path.join(input_dir, "image2_segmented.jpg")
results_seg.save(filename=output_path)

print(f"Instance segmentation completed. Output saved at: {output_path}")

