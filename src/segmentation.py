import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from ultralytics import YOLO
import torch
import numpy as np
import os

model = YOLO('/home/joshua/Snake_Gate/src/1115_seg2n.pt')
model((np.random.random((640, 640, 3))*255).astype(np.uint8))
print("Model loaded")

def process_image(input_path, output_path):

    cv_image = cv2.imread(input_path)
    if cv_image is None:
        print(f"Error: Unable to load image at {input_path}")
        return
        
    # Run YOLOv11 seg estimation
    h, w = cv_image.shape[:2]
    results = model(cv_image)
    # cv_image *= np.uint8(0)
    matplotlib_colors_bgr = [
        (180, 119, 31),   # C0 - tab:blue
        (14, 127, 255),   # C1 - tab:orange
        (44, 160, 44),    # C2 - tab:green
        (40, 39, 214),    # C3 - tab:red
        (189, 103, 148),  # C4 - tab:purple
        (75, 86, 140),    # C5 - tab:brown
        (194, 119, 227),  # C6 - tab:pink
        (127, 127, 127),  # C7 - tab:gray
        (34, 189, 188),   # C8 - tab:olive
        (207, 190, 23)    # C9 - tab:cyan
    ]

    # Process results
    for result in results:
        if result.masks is None:
            continue
        boxes = result.boxes.cpu().numpy()
        masks = result.masks.data.cpu().numpy()
        segments = result.masks.xy

        union_masks = torch.amax(result.masks.data, dim=0).cpu().numpy()
        print(union_masks.shape)

        # Draw bounding boxes and keypoints
        for idx, (box, mask, seg) in enumerate(zip(boxes, masks, segments)):
            # mask = union_masks
            mask = cv2.resize(mask, (w, h))
            # mask *= 0; cv2.fillPoly(mask, [seg.astype(np.int32)], 255)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            color = matplotlib_colors_bgr[idx]
            overlay = (mask[..., None].astype(np.uint8) * color).astype(np.uint8)
            cv2.addWeighted(cv_image, 1, overlay, 0.5, 0, cv_image)

            cv2.polylines(cv_image, [seg.astype(np.int32)], False, (0, 0, 255), 1)
            # break

        
        m = np.zeros(masks[0].shape, dtype=np.uint8)
        for idx, (box, mask) in enumerate(zip(boxes, masks)):
            m |= (mask.astype(np.uint8) << np.uint8(7-idx))

        cv2.imwrite(output_path, m)

        # Increment the counter for unique filenames

        # Publish the processed image
        # msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        # msg.header.stamp = rospy.Time.now()
        # self.image_pub.publish(msg)

        
def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            print(f"Processing {input_path}...")
            process_image(input_path, output_path)
    print(f"Processing complete. Images saved in {output_folder}.")

# Example usage
input_folder = "/home/joshua/Snake_Gate/dataset_seg_241231/images/val"
output_folder = "/home/joshua/masks1"
image_width = 848  # Example image width
image_height = 800  # Example image height

process_folder(input_folder, output_folder)


