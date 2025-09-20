import os
import cv2
import numpy as np

def undistort_images_in_folder(input_folder, output_folder, camera_matrix, dist_coeffs):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    TARGET_WIDTH = 848
    TARGET_HEIGHT = 800

    # Iterate through all images in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Read the original image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error reading {input_path}. Skipping...")
                continue

            # Get dimensions of the original image
            h, w = image.shape[:2]

            # Compute the optimal new camera matrix for the original size
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )

            # Undistort at original resolution
            undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

            # Optionally crop using the ROI (Region Of Interest) returned
            x, y, w_roi, h_roi = roi
            undistorted = undistorted[y:y+h_roi, x:x+w_roi]

            # Now resize to the target dimensions (848x800)
            final_undistorted = cv2.resize(undistorted, (TARGET_WIDTH, TARGET_HEIGHT))

            # Save the undistorted image
            cv2.imwrite(output_path, final_undistorted)
            print(f"Undistorted: {input_path} -> {output_path}")

    print(f"Undistortion complete. Images saved in {output_folder}.")

# Example usage
input_folder = "/home/joshua/Snake_Gate/masks_20241224/r3w1d3_2"  # Folder containing distorted images
output_folder = "/home/joshua/Snake_Gate/masks_20241224/r3w1d3_2_undistorted"  # Folder to save undistorted images

# Example camera matrix and distortion coefficients (replace with your calibration data)
camera_matrix = np.array([[285.8460998535156, 0.0, 418.7644958496094], [0.0, 286.0205993652344, 415.0235900878906], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([-0.009076775051653385, 0.043123308569192886, -0.03921201080083847, 0.006694579962641001], dtype=np.float32)

undistort_images_in_folder(input_folder, output_folder, camera_matrix, dist_coeffs)
