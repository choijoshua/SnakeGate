import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from ultralytics import YOLO
import torch
import numpy as np

class YOLOSegEstimator:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('yolo_seg_estimator', anonymous=True)

        # Load YOLOv11-seg model
        # self.model = YOLO('models/1115_seg2n.pt')
        self.model = YOLO('/home/joshua/Snake_Gate/src/1115_seg2n.pt')
        self.model((np.random.random((640, 640, 3))*255).astype(np.uint8))
        print("Model loaded")

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribe to image topic
        # self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        # self.image_sub = rospy.Subscriber('/camera/fisheye2/image_raw/compressed', CompressedImage, self.compressed_image_callback)
        self.image_sub = rospy.Subscriber('/a2rl/fisheye1/image_raw/compressed', CompressedImage, self.compressed_image_callback)

        self.image_pub = rospy.Publisher('/image', Image,queue_size=10)

        self.count = 0
        print("Ready")

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.process_image(cv_image)
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def compressed_image_callback(self, msg):
        try:
            # Convert compressed ROS image to OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            print(cv_image.shape, cv_image.dtype)
            self.process_image(cv_image)
        except Exception as e:
            rospy.logerr(f"Error processing compressed image: {str(e)}")

    def process_image(self, cv_image):
        # Run YOLOv11 seg estimation
        h, w = cv_image.shape[:2]
        results = self.model(cv_image)
        # cv_image *= np.uint8(0)
        image = cv_image
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
            
            self.count += 1
            cv2.imwrite("/home/joshua/masks1/{:05d}.png".format(self.count), m)
            image = m

            # Increment the counter for unique filenames

        # Publish the processed image
        # msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        # msg.header.stamp = rospy.Time.now()
        # self.image_pub.publish(msg)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, encoding="mono8"))
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        seg_estimator = YOLOSegEstimator()
        seg_estimator.run()
    except rospy.ROSInterruptException:
        pass