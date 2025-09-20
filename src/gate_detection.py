import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
import os
import numpy as np
import rospy
from collections import deque

class CornerDetection:
    def __init__(self):
        rospy.init_node('corner_detection', anonymous=True)

        self.bridge = CvBridge()

        self.mask_queue = deque(maxlen=10)
        self.image_queue = deque(maxlen=10)

        rospy.Subscriber('/image', Image, self.mask_callback)
        rospy.Subscriber('/a2rl/fisheye1/image_raw/compressed', CompressedImage, self.image_callback)

        self.image_pub = rospy.Publisher('/corner_detection_image', Image, queue_size=10)

    def mask_callback(self, msg):
        self.mask_queue.append(msg)
        self.try_process()

    def image_callback(self, msg):
        self.image_queue.append(msg)
        self.try_process()

    def try_process(self):
        if self.mask_queue and self.image_queue:
            mask_msg = self.mask_queue.popleft()
            image_msg = self.image_queue.popleft()
            self.process_synchronized(mask_msg, image_msg)

    def process_synchronized(self, mask_msg, image_msg):
        rospy.loginfo("Processing synchronized messages")
        try:
            # Convert the binary mask to an OpenCV format
            mask = self.bridge.imgmsg_to_cv2(mask_msg, "mono8")

            # Convert the compressed image to an OpenCV format
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Perform corner detection using the binary mask on the original image
            self.detect_gate_corners(mask, original_image)
        except Exception as e:
            rospy.logerr(f"Error processing synchronized messages: {str(e)}")

    def detect_gate_corners(self, image, original):
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("HEIEIEIIE")
        gray = image
        # Preprocess the image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)  # Edge detection

        # Morphological operations to clean up edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        _, binary_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        if num_labels == 2:
            num_labels, labels = cv2.connectedComponents(binary_mask)
        else:
            num_labels, labels = cv2.connectedComponents(gray)
        
        for label_id in range(1, num_labels):
            # Extract this component’s region
            component_mask = (labels == label_id).astype(np.uint8) * 255
            # cv2.imwrite(output_path, component_mask)
            # return

            # Find contours
            contours, _ = cv2.findContours(component_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                print(f"No contours found.")
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
                cv2.drawContours(original, [largest_contour], -1, (0, 255, 0), 2)  # Green for outer

                # Draw the inner contour line
                cv2.drawContours(original, [best_inner_contour], -1, (255, 0, 255), 2)  # Magenta for inner

                # Draw corners for visualization
                for corner in outer_corners:
                    cv2.circle(original, tuple(corner), 5, (0, 0, 255), -1)  # Red for outer corners
                for corner in inner_corners:
                    cv2.circle(original, tuple(corner), 5, (255, 0, 0), -1)  # Blue for inner corners
            else:
                print("Insufficient contours for inner corner selection.")
                return
        
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(original))

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try: 
        corner_detector = CornerDetection()
        corner_detector.run()
    except rospy.ROSInterruptException:
        pass



