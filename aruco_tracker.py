#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import yaml
import csv
import os
import threading

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped, PoseStamped, Quaternion, Pose

# Import the robot_control_modules
import robot_control_modules as rcm

class ArucoTracker:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('aruco_tracker_node', anonymous=True)

        # Parameters
        self.image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/color/camera_info')
        self.marker_size = rospy.get_param('~marker_size', 0.1)  # meters
        self.marker_id = rospy.get_param('~marker_id', 100)
        self.reference_frame = rospy.get_param('~reference_frame', 'camera_link')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_depth_optical_frame')
        self.marker_frame = rospy.get_param('~marker_frame', 'camera_marker')
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'j2n6s300_link_base')
        self.end_effector_frame = rospy.get_param('~end_effector_frame', 'j2n6s300_end_effector')
        self.arm_prefix = rospy.get_param('~arm_prefix', 'j2n6s300_')

        # Initialize variables
        self.camera_matrix = None
        self.dist_coeffs = None
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Setup ArUco
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)  # Original dictionary
        self.aruco_params = aruco.DetectorParameters_create()

        # Subscribers
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(self.image_topic, Image, self.image_callback)

        # Publisher for target pose visualization
        self.target_pose_pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)

        # Wait until camera info is received
        rospy.loginfo("Waiting for camera info...")
        while self.camera_matrix is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Camera info received.")

        # Store the last detected tag and end-effector positions in base frame
        self.last_tag_pose_base = None
        self.last_ee_pose_base = None

        # Create OpenCV window
        cv2.namedWindow("Aruco Tracker", cv2.WINDOW_NORMAL)

        # Initialize home pose
        self.home_pose = None
        self.home_pose_file = 'home_pose.yaml'
        self.load_home_pose()

        # Initialize log file
        self.log_file = 'pose_log.csv'
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                writer = csv.writer(f)
                # Write headers
                writer.writerow(['timestamp',
                                 'ee_pos_x', 'ee_pos_y', 'ee_pos_z',
                                 'ee_orient_x', 'ee_orient_y', 'ee_orient_z', 'ee_orient_w',
                                 'tag_pos_x', 'tag_pos_y', 'tag_pos_z',
                                 'tag_orient_x', 'tag_orient_y', 'tag_orient_z', 'tag_orient_w'])

        # For virtual sphere
        self.sphere_radius = rospy.get_param('~sphere_radius', 0.1)  # meters

        # Initialize image variable
        self.cv_image = None

        # Flags for movement and logging
        self.is_moving = False
        self.is_logging = False
        self.move_thread = None

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape((3, 3))
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera matrix and distortion coefficients received.")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            # Estimate pose of each marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.marker_id:
                    # Draw marker
                    aruco.drawDetectedMarkers(cv_image, corners, ids)

                    # Draw axis
                    aruco.drawAxis(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.05)

                    # 3D coordinates in camera frame
                    tvec = tvecs[i].flatten()
                    cv2.putText(cv_image, f"Camera Frame: X={tvec[0]:.2f} Y={tvec[1]:.2f} Z={tvec[2]:.2f} m",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Create a TransformStamped message from camera to marker
                    marker_transform = TransformStamped()
                    marker_transform.header.stamp = rospy.Time.now()
                    marker_transform.header.frame_id = self.camera_frame
                    marker_transform.child_frame_id = self.marker_frame
                    marker_transform.transform.translation.x = tvec[0]
                    marker_transform.transform.translation.y = tvec[1]
                    marker_transform.transform.translation.z = tvec[2]

                    # Convert rotation vector to quaternion
                    rmat, _ = cv2.Rodrigues(rvecs[i])
                    quat = self.rotation_matrix_to_quaternion(rmat)
                    marker_transform.transform.rotation.x = quat[0]
                    marker_transform.transform.rotation.y = quat[1]
                    marker_transform.transform.rotation.z = quat[2]
                    marker_transform.transform.rotation.w = quat[3]

                    try:
                        # Lookup transform from camera to robot base
                        transform_cam_to_base = self.tf_buffer.lookup_transform(
                            self.robot_base_frame,
                            self.camera_frame,
                            rospy.Time(0),
                            rospy.Duration(1.0)
                        )

                        # Convert marker pose to geometry_msgs.PoseStamped
                        marker_pose_camera = self.transform_stamped_to_pose_stamped(marker_transform)
                        marker_pose_base = tf2_geometry_msgs.do_transform_pose(
                            marker_pose_camera,
                            transform_cam_to_base
                        )

                        # Extract position in base frame
                        base_x = marker_pose_base.pose.position.x
                        base_y = marker_pose_base.pose.position.y
                        base_z = marker_pose_base.pose.position.z

                        cv2.putText(cv_image, f"Base Frame: X={base_x:.2f} Y={base_y:.2f} Z={base_z:.2f} m",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        # Store the last detected tag pose in base frame
                        self.last_tag_pose_base = marker_pose_base.pose

                        # Get end-effector pose in base frame
                        transform_base_to_ee = self.tf_buffer.lookup_transform(
                            self.robot_base_frame,
                            self.end_effector_frame,
                            rospy.Time(0),
                            rospy.Duration(1.0)
                        )

                        ee_x = transform_base_to_ee.transform.translation.x
                        ee_y = transform_base_to_ee.transform.translation.y
                        ee_z = transform_base_to_ee.transform.translation.z

                        cv2.putText(cv_image, f"End Effector: X={ee_x:.2f} Y={ee_y:.2f} Z={ee_z:.2f} m",
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        # Store the last end-effector pose
                        self.last_ee_pose_base = Pose()
                        self.last_ee_pose_base.position = transform_base_to_ee.transform.translation
                        self.last_ee_pose_base.orientation = transform_base_to_ee.transform.rotation

                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        rospy.logerr(f"TF2 Lookup Error: {e}")

        # Update the image for display
        self.cv_image = cv_image

    def rotation_matrix_to_quaternion(self, R):
        """
        Convert a rotation matrix to a quaternion.
        """
        q = np.empty((4, ))
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[3] = 0.25 / s
            q[0] = (R[2,1] - R[1,2]) * s
            q[1] = (R[0,2] - R[2,0]) * s
            q[2] = (R[1,0] - R[0,1]) * s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                q[3] = (R[2,1] - R[1,2]) / s
                q[0] = 0.25 * s
                q[1] = (R[0,1] + R[1,0]) / s
                q[2] = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                q[3] = (R[0,2] - R[2,0]) / s
                q[0] = (R[0,1] + R[1,0]) / s
                q[1] = 0.25 * s
                q[2] = (R[1,2] + R[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                q[3] = (R[1,0] - R[0,1]) / s
                q[0] = (R[0,2] + R[2,0]) / s
                q[1] = (R[1,2] + R[2,1]) / s
                q[2] = 0.25 * s
        return q

    def transform_stamped_to_pose_stamped(self, transform_stamped):
        """
        Convert a TransformStamped message to a PoseStamped message.
        """
        pose_stamped = PoseStamped()
        pose_stamped.header = transform_stamped.header
        pose_stamped.pose.position = transform_stamped.transform.translation
        pose_stamped.pose.orientation = transform_stamped.transform.rotation
        return pose_stamped

    def load_home_pose(self):
        try:
            with open(self.home_pose_file, 'r') as f:
                data = yaml.safe_load(f)
                if data:
                    self.home_pose = Pose()
                    self.home_pose.position.x = data['position']['x']
                    self.home_pose.position.y = data['position']['y']
                    self.home_pose.position.z = data['position']['z']
                    self.home_pose.orientation.x = data['orientation']['x']
                    self.home_pose.orientation.y = data['orientation']['y']
                    self.home_pose.orientation.z = data['orientation']['z']
                    self.home_pose.orientation.w = data['orientation']['w']
                    rospy.loginfo("Home pose loaded from file.")
        except FileNotFoundError:
            rospy.logwarn("Home pose file not found. No home pose loaded.")

    def save_home_pose(self):
        if self.home_pose:
            data = {
                'position': {
                    'x': self.home_pose.position.x,
                    'y': self.home_pose.position.y,
                    'z': self.home_pose.position.z
                },
                'orientation': {
                    'x': self.home_pose.orientation.x,
                    'y': self.home_pose.orientation.y,
                    'z': self.home_pose.orientation.z,
                    'w': self.home_pose.orientation.w
                }
            }
            with open(self.home_pose_file, 'w') as f:
                yaml.dump(data, f)
            rospy.loginfo("Home pose saved to file.")
        else:
            rospy.logwarn("No home pose to save.")

    def set_home_pose(self):
        try:
            # Get current end-effector pose
            transform_base_to_ee = self.tf_buffer.lookup_transform(
                self.robot_base_frame,
                self.end_effector_frame,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            self.home_pose = Pose()
            self.home_pose.position = transform_base_to_ee.transform.translation
            self.home_pose.orientation = transform_base_to_ee.transform.rotation

            self.save_home_pose()
            rospy.loginfo("Current end-effector pose set as home pose.")
        except Exception as e:
            rospy.logerr(f"Failed to get current end-effector pose: {e}")

    def move_to_home_pose(self):
        if self.home_pose:
            # Prepare position and orientation arrays
            position = [self.home_pose.position.x, self.home_pose.position.y, self.home_pose.position.z]
            orientation = [self.home_pose.orientation.x, self.home_pose.orientation.y, self.home_pose.orientation.z, self.home_pose.orientation.w]

            rospy.loginfo("Moving to home pose...")
            result = rcm.cartesian_pose_client(position, orientation, self.arm_prefix)
            if result:
                rospy.loginfo("Moved to home pose successfully.")
            else:
                rospy.logwarn("Failed to move to home pose.")
        else:
            rospy.logwarn("Home pose is not set.")

    def log_current_data(self):
        if self.last_tag_pose_base and self.last_ee_pose_base:
            with open(self.log_file, 'a') as f:
                writer = csv.writer(f)
                timestamp = rospy.Time.now().to_sec()
                ee_pos = self.last_ee_pose_base.position
                ee_orient = self.last_ee_pose_base.orientation
                tag_pos = self.last_tag_pose_base.position
                tag_orient = self.last_tag_pose_base.orientation
                writer.writerow([timestamp,
                                 ee_pos.x, ee_pos.y, ee_pos.z,
                                 ee_orient.x, ee_orient.y, ee_orient.z, ee_orient.w,
                                 tag_pos.x, tag_pos.y, tag_pos.z,
                                 tag_orient.x, tag_orient.y, tag_orient.z, tag_orient.w])
            rospy.loginfo("Current data logged.")
        else:
            rospy.logwarn("Cannot log data. Missing end-effector or tag pose.")

    def move_to_last_tag_pose(self):
        """
        Move the robot's end-effector to a position relative to the last detected ArUco tag using robot_control_modules.cartesian_pose_client.
        """
        if self.last_tag_pose_base is None:
            rospy.logwarn("No tag pose available to move to.")
            return

        target_pose = Pose()
        target_pose.position = self.last_tag_pose_base.position
        target_pose.orientation = self.last_tag_pose_base.orientation

        # Apply offsets: e.g., 5 cm upwards (Z-axis) and 3 cm forward (X-axis)
        offset_x = 0.03  # 3 cm forward
        offset_z = 0.05  # 5 cm upwards

        target_pose.position.x += offset_x
        target_pose.position.z += offset_z

        # For orientation, maintain the current orientation or set a desired one
        try:
            # Get the current end-effector orientation
            transform_base_to_ee = self.tf_buffer.lookup_transform(
                self.robot_base_frame,
                self.end_effector_frame,
                rospy.Time(0),
                rospy.Duration(1.0)
            )

            # Use the current end-effector orientation
            target_pose.orientation = transform_base_to_ee.transform.rotation

            rospy.loginfo("Current end-effector orientation obtained and maintained.")
        except Exception as e:
            rospy.logerr(f"Failed to get current end-effector orientation: {e}")
            return

        # Log the target pose
        rospy.loginfo(f"Target Pose: Position -> X: {target_pose.position.x:.2f}, Y: {target_pose.position.y:.2f}, Z: {target_pose.position.z:.2f}")
        rospy.loginfo(f"Target Pose: Orientation -> X: {target_pose.orientation.x:.2f}, Y: {target_pose.orientation.y:.2f}, Z: {target_pose.orientation.z:.2f}, W: {target_pose.orientation.w:.2f}")

        # Publish the target pose for visualization
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.robot_base_frame
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose = target_pose
        self.target_pose_pub.publish(pose_stamped)
        rospy.loginfo("Target pose with offsets published for visualization.")

        # Prepare position and orientation arrays
        position = [target_pose.position.x, target_pose.position.y, target_pose.position.z]
        orientation = [target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w]

        # Start movement in a separate thread
        self.is_moving = True
        self.is_logging = True
        self.move_thread = threading.Thread(target=self.execute_movement, args=(position, orientation))
        self.move_thread.start()

    def execute_movement(self, position, orientation):
        rospy.loginfo("Starting movement to target pose...")
        result = rcm.cartesian_pose_client(position, orientation, self.arm_prefix)

        if result:
            rospy.loginfo("Movement to target pose completed successfully.")
        else:
            rospy.logwarn("Movement to target pose failed.")

        # Check if end-effector is within virtual sphere
        self.check_virtual_sphere()

        # Move back to home pose
        self.move_to_home_pose()

        # Movement completed
        self.is_moving = False
        self.is_logging = False

    def check_virtual_sphere(self):
        rospy.loginfo("Checking if end-effector is within virtual sphere...")
        if self.last_tag_pose_base and self.last_ee_pose_base:
            dx = self.last_ee_pose_base.position.x - self.last_tag_pose_base.position.x
            dy = self.last_ee_pose_base.position.y - self.last_tag_pose_base.position.y
            dz = self.last_ee_pose_base.position.z - self.last_tag_pose_base.position.z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            if distance < self.sphere_radius:
                rospy.loginfo("End-effector is within virtual sphere. Marking success.")
            else:
                rospy.logwarn("End-effector is not within virtual sphere.")
        else:
            rospy.logwarn("Cannot check virtual sphere. Missing end-effector or tag pose.")

    def run(self):
        rospy.loginfo("Aruco Tracker is running. Press 'g' in the OpenCV window to move the robot to the tag's location with offsets and log data.")
        rospy.loginfo("Press 'h' to set current end-effector pose as home pose.")
        rospy.loginfo("Press 'm' to move to home pose.")
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.cv_image is not None:
                cv2.imshow("Aruco Tracker", self.cv_image)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('g'):
                    rospy.loginfo("Key 'g' pressed. Initiating movement to tag pose.")
                    self.move_to_last_tag_pose()
                elif key == ord('h'):
                    rospy.loginfo("Key 'h' pressed. Setting home pose.")
                    self.set_home_pose()
                elif key == ord('m'):
                    rospy.loginfo("Key 'm' pressed. Moving to home pose.")
                    self.move_to_home_pose()

            # If robot is moving, log data
            if self.is_logging:
                self.log_current_data()

            rate.sleep()

        # Cleanup
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        tracker = ArucoTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass
