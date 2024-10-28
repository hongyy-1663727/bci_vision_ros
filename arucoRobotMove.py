#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander
import tf

class ArucoDetector:
    def __init__(self):
        rospy.init_node('aruco_tag_detector', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)  # Original ArUco Dictionary
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.move_group = MoveGroupCommander('arm')  # Specify the Kinova arm planning group
        self.listener = tf.TransformListener()
        self.tag_size = 0.1  # Size of the tag in meters (100 mm)
        self.latest_image = None
        self.detected_pose = None
        print("Press 'p' to move the robot to the ArUco tag's center.")

    def image_callback(self, data):
        # Convert the ROS Image message to OpenCV format
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            rospy.logerr("Could not convert image: %s", e)

    def detect_aruco(self):
        if self.latest_image is None:
            return

        # Convert image to grayscale
        gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the image
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        # Get end-effector current position
        try:
            (trans, rot) = self.listener.lookupTransform('j2n6s300_link_base', 'j2n6s300_end_effector', rospy.Time(0))
            ee_position = trans  # End-effector position
            ee_orientation = rot  # End-effector orientation
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Could not get end-effector pose")
            ee_position = [0.0, 0.0, 0.0]

        if ids is not None:
            # Assume we're interested in the first detected tag only
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.tag_size, np.eye(3), np.zeros(5))

            # Extract translation and rotation
            tvec = tvec[0][0]
            self.detected_pose = PoseStamped()
            self.detected_pose.header.frame_id = "camera_color_optical_frame"
            self.detected_pose.pose.position.x = tvec[0]
            self.detected_pose.pose.position.y = tvec[1]
            self.detected_pose.pose.position.z = tvec[2]
            # rospy.loginfo(f"Detected ArUco tag at position: {tvec}")

            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(self.latest_image, corners, ids)
            # Draw axis to indicate pose
            cv2.aruco.drawAxis(self.latest_image, np.eye(3), np.zeros(5), rvec[0], tvec, 0.1)

            # Display the coordinates of the ArUco tag on the image window
            cv2.putText(self.latest_image, f"ArUco Position: x={tvec[0]:.2f}, y={tvec[1]:.2f}, z={tvec[2]:.2f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Get the current end-effector pose
            end_effector_pose = self.move_group.get_current_pose().pose
            cv2.putText(self.latest_image, f"End Effector: x={ee_position[0]:.2f}, y={ee_position[1]:.2f}, z={ee_position[2]:.2f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show image
        cv2.imshow('Aruco Tag Detection', self.latest_image)
        cv2.waitKey(1)

    def move_to_tag(self):
        if self.detected_pose is None:
            rospy.logwarn("No ArUco tag detected to move to.")
            return

        try:
            # Transform the pose of ArUco tag to base frame
            self.detected_pose.header.stamp = rospy.Time.now()
            transformed_pose = self.listener.transformPose('j2n6s300_link_base', self.detected_pose)

            # Set the target pose for the move group
            self.move_group.set_pose_target(transformed_pose.pose)
            success = self.move_group.go(wait=True)
            if success:
                rospy.loginfo("Robot moved to ArUco tag center successfully.")
            else:
                rospy.logerr("Failed to move the robot to the target pose.")

        except Exception as e:
            rospy.logerr("Error in transforming pose: %s", e)

    def run(self):
        while not rospy.is_shutdown():
            self.detect_aruco()
            if cv2.waitKey(1) & 0xFF == ord('p'):
                self.move_to_tag()

if __name__ == '__main__':
    detector = ArucoDetector()
    try:
        detector.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
