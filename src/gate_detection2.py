import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
import os
import numpy as np
import rospy

def detect_gate_corners(image_path, image_path2, output_path):
    # Load the image
    image = cv2.imread(image_path)
    image2 = cv2.imread(image_path2)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    if image2 is None:
        print(f"Error: Unable to load image at {image_path2}")
        return
    
    image = cv2.resize(image, (848, 800))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocess the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection

    # Morphological operations to clean up edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    _, binary_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(binary_mask)

    if num_labels == 2:
        _, binary_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(binary_mask)
    
    
    for label_id in range(1, num_labels):
        # Extract this component’s region
        component_mask = (labels == label_id).astype(np.uint8) * 255
        # cv2.imwrite(output_path, component_mask)
        # return

        # Find contours
        contours, _ = cv2.findContours(component_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"No contours found in {image_path}.")
            return

        # Sort all contours by area (descending)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # 1) Outer gate contour: take the largest (but also can iterate if you have multiple gates)
        largest_contour = contours[0]
        param = 0.02
        epsilon = param * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        done = 0
        while len(approx) != 4 and done < 30:
            if len(approx) > 4:
                param *= 1.1
            else:
                param *= 0.9
            epsilon = param * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            done += 1

        
        if len(approx) != 4:
            print(f"Outer contour not forming a quadrilateral. Found {len(approx)} corners instead.")
            rect = cv2.minAreaRect(largest_contour)
            corners = cv2.boxPoints(rect)
            outer_corners = np.int0(corners)
        else:
            outer_corners = approx.reshape(-1, 2)
            # Optional: could do a fallback like minAreaRect here
        # outer_corners = approx.reshape(-1, 2)

        # 2) Inner contour selection logic
        # If the drone gate has an inner opening, you might check subsequent contours for the "hole"
        # But your current code checks contours[2] & contours[3].
        if len(contours) > 1:
            inner_contour_candidates = contours[1:]  # next 4 largest after the outer
            # Or you could do something more systematic (e.g., find the largest contour *inside* the outer contour).
            
            # Example: pick the largest from those next few
            best_inner_contour = max(inner_contour_candidates, key=cv2.contourArea)
            
            param = 0.02
            epsilon_inner = param * cv2.arcLength(best_inner_contour, True)
            approx_inner = cv2.approxPolyDP(best_inner_contour, epsilon_inner, True)

            done = 0
            while len(approx_inner) != 4 and done < 30:
                if len(approx_inner) > 4:
                    param *= 1.1
                else:
                    param *= 0.9
                epsilon_inner = param * cv2.arcLength(best_inner_contour, True)
                approx_inner = cv2.approxPolyDP(best_inner_contour, epsilon_inner, True)
                done += 1

            inner_corners = approx_inner.reshape(-1, 2)

            # Draw the outer contour line
            # cv2.drawContours(image2, [largest_contour], -1, (0, 255, 0), 2)  # Green for outer

            # # Draw the inner contour line
            # cv2.drawContours(image2, [best_inner_contour], -1, (255, 0, 255), 2)  # Magenta for inner

            # Draw corners for visualization
            for corner in outer_corners:
                cv2.circle(image2, tuple(corner), 5, (0, 0, 255), -1)  # Red for outer corners
            for corner in inner_corners:
                cv2.circle(image2, tuple(corner), 5, (255, 0, 0), -1)  # Blue for inner corners
        else:
            print("Insufficient contours for inner corner selection.")
            cv2.imwrite(output_path, image2)
            return

    # Save the processed image
    cv2.imwrite(output_path, image)


def process_folder(input_folder, input_folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, file_name)
            input_path2 = os.path.join(input_folder2, file_name)
            output_path = os.path.join(output_folder, file_name)
            print(f"Processing {input_path}...")
            detect_gate_corners(input_path, input_path2, output_path)
    print(f"Processing complete. Images saved in {output_folder}.")


# Define paths
# input_folder = "/home/joshua/Snake_Gate/masks_20241224/r3w1d3_2"
input_folder = "/home/joshua/masks1"
input_folder2 = "/home/joshua/Snake_Gate/dataset_seg_241231/images/val"
output_folder = "/home/joshua/masks2"

# Process all images in the folder
process_folder(input_folder, input_folder2, output_folder)