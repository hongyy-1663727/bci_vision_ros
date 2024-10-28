#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import tf
import actionlib
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from kinova_msgs.msg import ArmPoseAction, ArmPoseGoal
from threading import Event

class ArucoRobotController:
    def __init__(self):
        rospy.init_node('aruco_robot_controller', anonymous=True)

        # Initialize variables
        self.camera_matrix = None
        self.dist_coeffs = None
        self.marker_length = 0.05  # Adjust to your marker's size in meters
        self.camera_info_received = False
        self.move_robot_event = Event()

        # Transformation from camera frame to robot base frame (replace with your values)
        self.translation = np.array([-0.1839708648988643, -0.6324757513256634, 0.1409081406428656])
        self.rotation_quaternion = np.array([-0.39441502940037465, -0.15800439346688022, -0.142903810072808, 0.893895909653322])
        self.transformation_matrix = self.get_transformation_matrix()

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize action client for Kinova robot
        self.client = actionlib.SimpleActionClient('/j2n6s300_driver/pose_action/tool_pose', ArmPoseAction)
        rospy.loginfo("Waiting for action server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to action server.")

        # Subscribe to camera topics
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        # Initialize TF listener
        self.tf_listener = tf.TransformListener()

    def get_transformation_matrix(self):
        # Converts the rotation quaternion to a rotation matrix
        rotation_matrix = tf.transformations.quaternion_matrix(self.rotation_quaternion)
        transformation_matrix = np.identity(4)
        transformation_matrix[0:3, 0:3] = rotation_matrix[0:3, 0:3]
        transformation_matrix[0:3, 3] = self.translation
        return transformation_matrix

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            # Retrieves the camera intrinsic parameters
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            self.camera_info_received = True
            rospy.loginfo("Camera calibration parameters received.")

    def image_callback(self, msg):
        if not self.camera_info_received:
            rospy.logwarn("Waiting for camera calibration parameters...")
            return

        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Get end-effector current position
        try:
            (trans, rot) = self.tf_listener.lookupTransform('j2n6s300_link_base', 'j2n6s300_end_effector', rospy.Time(0))
            ee_position = trans  # End-effector position
            ee_orientation = rot  # End-effector orientation
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Could not get end-effector pose")
            ee_position = [0.0, 0.0, 0.0]

        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

            # Estimate pose of each marker
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

            for rvec, tvec in zip(rvecs, tvecs):
                # Draw axis for the marker
                cv2.aruco.drawAxis(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)

                # Convert rotation vector to rotation matrix
                rotation_matrix_marker, _ = cv2.Rodrigues(rvec[0])

                # Apply rotation to align frames
                R_x_neg_90 = np.array([
                    [1, 0,  0],
                    [0, 0,  1],
                    [0, -1, 0]
                ])

                # Adjust rotation matrix and translation vector
                rotation_matrix_marker = np.dot(R_x_neg_90, rotation_matrix_marker)
                tvec_adjusted = np.dot(R_x_neg_90, tvec[0].reshape(3, 1)).flatten()

                # Create homogeneous transformation matrix for the marker in camera frame
                transformation_marker_cam = np.identity(4)
                transformation_marker_cam[0:3, 0:3] = rotation_matrix_marker
                transformation_marker_cam[0:3, 3] = tvec_adjusted

                # Transform marker pose to robot base frame
                transformation_marker_base = np.dot(self.transformation_matrix, transformation_marker_cam)

                # Extract position and orientation
                position = transformation_marker_base[0:3, 3]
                quaternion = tf.transformations.quaternion_from_matrix(transformation_marker_base)

                # Display the position in camera frame
                cv2.putText(cv_image,
                            f"Cam Frame Pos: x={tvec_adjusted[0]:.2f}, y={tvec_adjusted[1]:.2f}, z={tvec_adjusted[2]:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Display the position in robot base frame
                cv2.putText(cv_image,
                            f"Robot Base Pos: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Display the current end-effector position
                cv2.putText(cv_image,
                            f"EE Pos: x={ee_position[0]:.2f}, y={ee_position[1]:.2f}, z={ee_position[2]:.2f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Show the image
                cv2.imshow("ArUco Detection", cv_image)
                key = cv2.waitKey(1)

                # Check if 'g' key is pressed to move the robot
                if key == ord('g'):
                    self.move_robot_event.set()

                if self.move_robot_event.is_set():
                    # Create PoseStamped message
                    pose = PoseStamped()
                    pose.header.stamp = rospy.Time.now()
                    pose.header.frame_id = 'j2n6s300_link_base'
                    pose.pose.position.x = position[0]
                    pose.pose.position.y = position[1]
                    pose.pose.position.z = position[2]
                    pose.pose.orientation.x = quaternion[0]
                    pose.pose.orientation.y = quaternion[1]
                    pose.pose.orientation.z = quaternion[2]
                    pose.pose.orientation.w = quaternion[3]

                    # Move robot to the marker's pose
                    self.move_robot_to_pose(pose)
                    self.move_robot_event.clear()  # Reset the event for the next detection

                break  # Remove this if you want to process all detected markers
        else:
            # Display the current end-effector position even if no markers are detected
            cv2.putText(cv_image,
                        f"EE Pos: x={ee_position[0]:.2f}, y={ee_position[1]:.2f}, z={ee_position[2]:.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Show the image
            cv2.imshow("ArUco Detection", cv_image)
            cv2.waitKey(1)

    def move_robot_to_pose(self, pose):
        goal = ArmPoseGoal()
        goal.pose = pose
        self.client.send_goal(goal)
        self.client.wait_for_result()
        rospy.loginfo("Moved robot to the marker's position.")

    def shutdown(self):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        controller = ArucoRobotController()
        rospy.on_shutdown(controller.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
