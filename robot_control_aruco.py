#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os

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
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        self.marker_frame = rospy.get_param('~marker_frame', 'camera_marker')
        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'j2n6s300_link_base')
        self.end_effector_frame = rospy.get_param('~end_effector_frame', 'j2n6s300_end_effector')
        self.arm_prefix = rospy.get_param('~arm_prefix', 'j2n6s300_')
        self.sphere_radius = rospy.get_param('~sphere_radius', 0.1)  # meters

        # Initialize variables
        self.camera_matrix = None
        self.dist_coeffs = None
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Setup ArUco
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = aruco.DetectorParameters_create()

        # Subscribers
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(self.image_topic, Image, self.image_callback)

        # Publisher for target pose visualization
        self.target_pose_pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)

        # Initialize home pose
        self.home_pose = None
        self.load_home_pose()

        # Open log file
        self.log_file = open('movement_log.csv', 'w')
        self.log_file.write('timestamp,ee_x,ee_y,ee_z,tag_x,tag_y,tag_z,sphere_radius\n')

        # Wait until camera info is received
        rospy.loginfo("Waiting for camera info...")
        while self.camera_matrix is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Camera info received.")

        # Store the last detected tag position in base frame
        self.last_tag_pose_base = None

        # Create OpenCV window
        cv2.namedWindow("Aruco Tracker", cv2.WINDOW_NORMAL)

    def load_home_pose(self):
        # Load home pose from file if exists
        home_pose_file = 'home_pose.json'
        if os.path.exists(home_pose_file):
            with open(home_pose_file, 'r') as f:
                pose_data = json.load(f)
                self.home_pose = Pose()
                self.home_pose.position.x = pose_data['position']['x']
                self.home_pose.position.y = pose_data['position']['y']
                self.home_pose.position.z = pose_data['position']['z']
                self.home_pose.orientation.x = pose_data['orientation']['x']
                self.home_pose.orientation.y = pose_data['orientation']['y']
                self.home_pose.orientation.z = pose_data['orientation']['z']
                self.home_pose.orientation.w = pose_data['orientation']['w']
                rospy.loginfo("Home pose loaded from file.")
        else:
            rospy.loginfo("No home pose file found. Home pose is not set.")

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

                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        rospy.logerr(f"TF2 Lookup Error: {e}")

        # Display the image
        cv2.imshow("Aruco Tracker", cv_image)
        # Capture key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            rospy.loginfo("Key 'g' pressed.")
            self.move_to_last_tag_pose()
        elif key == ord('h'):
            rospy.loginfo("Key 'h' pressed.")
            self.set_home_pose()

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

    def set_home_pose(self):
        """
        Set the current end-effector pose as the home pose.
        """
        try:
            # Get the current end-effector pose in base frame
            transform_base_to_ee = self.tf_buffer.lookup_transform(
                self.robot_base_frame,
                self.end_effector_frame,
                rospy.Time(0),
                rospy.Duration(1.0)
            )

            self.home_pose = Pose()
            self.home_pose.position = transform_base_to_ee.transform.translation
            self.home_pose.orientation = transform_base_to_ee.transform.rotation

            # Save to file
            home_pose_file = 'home_pose.json'
            with open(home_pose_file, 'w') as f:
                pose_data = {
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
                json.dump(pose_data, f)
                rospy.loginfo("Home pose saved to file.")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF2 Lookup Error when setting home pose: {e}")

    def move_to_home_pose(self):
        """
        Move the robot's end-effector to the home pose.
        """
        if self.home_pose is None:
            rospy.logwarn("Home pose is not set.")
            return

        # Prepare position and orientation arrays
        position = [self.home_pose.position.x, self.home_pose.position.y, self.home_pose.position.z]
        orientation = [self.home_pose.orientation.x, self.home_pose.orientation.y, self.home_pose.orientation.z, self.home_pose.orientation.w]

        # Use the robot_control_modules's cartesian_pose_client
        rospy.loginfo("Returning to home pose...")
        result = rcm.cartesian_pose_client(position, orientation, self.arm_prefix)

        if result:
            rospy.loginfo("Robot has returned to home pose.")
        else:
            rospy.logwarn("Failed to return to home pose.")

    def log_movement(self, ee_position, tag_position):
        """
        Log the end-effector and tag positions to the log file.
        """
        timestamp = rospy.Time.now().to_sec()
        self.log_file.write(f"{timestamp},{ee_position.x},{ee_position.y},{ee_position.z},{tag_position.x},{tag_position.y},{tag_position.z},{self.sphere_radius}\n")
        self.log_file.flush()
        rospy.loginfo("Movement logged.")

    def move_to_last_tag_pose(self):
        """
        Move the robot's end-effector to the virtual sphere around the ArUco tag.
        """
        if self.last_tag_pose_base is None:
            rospy.logwarn("No tag pose available to move to.")
            return

        try:
            # Get the current end-effector pose in base frame
            transform_base_to_ee = self.tf_buffer.lookup_transform(
                self.robot_base_frame,
                self.end_effector_frame,
                rospy.Time(0),
                rospy.Duration(1.0)
            )

            ee_position = transform_base_to_ee.transform.translation

            # Compute vector from end-effector to tag
            dx = self.last_tag_pose_base.position.x - ee_position.x
            dy = self.last_tag_pose_base.position.y - ee_position.y
            dz = self.last_tag_pose_base.position.z - ee_position.z

            distance = np.sqrt(dx**2 + dy**2 + dz**2)

            if distance <= self.sphere_radius:
                rospy.loginfo("End-effector is already within the virtual sphere.")
                # Log movement and return to home
                self.log_movement(ee_position, self.last_tag_pose_base.position)
                self.move_to_home_pose()
                return

            # Compute the point on the line at sphere_radius from the tag
            ratio = self.sphere_radius / distance
            target_x = self.last_tag_pose_base.position.x - dx * ratio
            target_y = self.last_tag_pose_base.position.y - dy * ratio
            target_z = self.last_tag_pose_base.position.z - dz * ratio

            target_pose = Pose()
            target_pose.position.x = target_x
            target_pose.position.y = target_y
            target_pose.position.z = target_z

            # Keep the orientation of the tag or set a desired orientation
            target_pose.orientation = self.last_tag_pose_base.orientation

            # Log the target pose
            rospy.loginfo(f"Target Pose: Position -> X: {target_pose.position.x:.3f}, Y: {target_pose.position.y:.3f}, Z: {target_pose.position.z:.3f}")
            rospy.loginfo(f"Target Pose: Orientation -> X: {target_pose.orientation.x:.3f}, Y: {target_pose.orientation.y:.3f}, Z: {target_pose.orientation.z:.3f}, W: {target_pose.orientation.w:.3f}")

            # Publish the target pose for visualization
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.robot_base_frame
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.pose = target_pose
            self.target_pose_pub.publish(pose_stamped)
            rospy.loginfo("Target pose published for visualization.")

            # Prepare position and orientation arrays
            position = [target_pose.position.x, target_pose.position.y, target_pose.position.z]
            orientation = [target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w]

            # Use the robot_control_modules's cartesian_pose_client
            rospy.loginfo("Sending goal to cartesian_pose_client...")
            result = rcm.cartesian_pose_client(position, orientation, self.arm_prefix)

            if result:
                rospy.loginfo("Motion execution completed successfully.")
                # Get updated end-effector pose
                transform_base_to_ee = self.tf_buffer.lookup_transform(
                    self.robot_base_frame,
                    self.end_effector_frame,
                    rospy.Time(0),
                    rospy.Duration(1.0)
                )
                ee_position = transform_base_to_ee.transform.translation

                # Compute new distance
                dx = self.last_tag_pose_base.position.x - ee_position.x
                dy = self.last_tag_pose_base.position.y - ee_position.y
                dz = self.last_tag_pose_base.position.z - ee_position.z
                distance = np.sqrt(dx**2 + dy**2 + dz**2)

                if distance <= self.sphere_radius:
                    rospy.loginfo("End-effector has reached the virtual sphere.")
                    # Log movement and return to home
                    self.log_movement(ee_position, self.last_tag_pose_base.position)
                    self.move_to_home_pose()
                else:
                    rospy.logwarn("End-effector did not reach the virtual sphere.")
                    # Log movement anyway
                    self.log_movement(ee_position, self.last_tag_pose_base.position)
            else:
                rospy.logwarn("Motion execution failed.")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF2 Lookup Error: {e}")

    def run(self):
        rospy.loginfo("Aruco Tracker is running. Press 'g' to move to the tag's location, 'h' to set home pose.")
        rate = rospy.Rate(10)  # 10 Hz
        try:
            while not rospy.is_shutdown():
                rate.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            if self.log_file:
                self.log_file.close()
                rospy.loginfo("Log file closed.")

if __name__ == '__main__':
    try:
        tracker = ArucoTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass
