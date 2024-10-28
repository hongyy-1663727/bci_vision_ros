#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
import threading

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped, PoseStamped, Quaternion, Pose

# Import the robot_control_modules
import robot_control_modules as rcm
from actionlib import SimpleActionClient
from kinova_msgs.msg import ArmPoseAction, ArmPoseGoal

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

        # Wait until camera info is received
        rospy.loginfo("Waiting for camera info...")
        while self.camera_matrix is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Camera info received.")

        # Initialize action client
        action_address = '/' + self.arm_prefix + 'driver/pose_action/tool_pose'
        self.arm_pose_client = SimpleActionClient(action_address, ArmPoseAction)
        rospy.loginfo("Waiting for the arm_pose_action server...")
        self.arm_pose_client.wait_for_server()
        rospy.loginfo("Connected to arm_pose_action server.")

        # Store the last detected tag position in base frame
        self.last_tag_pose_base = None
        self.current_target_pose = None  # Shared variable for target pose
        self.tracing_enabled = False

        # Thread control
        self.control_thread = None
        self.stop_thread = threading.Event()

        # Create OpenCV window
        cv2.namedWindow("Aruco Tracker", cv2.WINDOW_NORMAL)

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

        tag_detected = False

        if ids is not None:
            # Estimate pose of each marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.marker_id:
                    tag_detected = True
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

                        # Update current target pose if tracing is enabled
                        if self.tracing_enabled:
                            self.update_current_target_pose()

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

        else:
            # No markers detected
            pass

        # Display the image
        cv2.imshow("Aruco Tracker", cv_image)
        # Capture key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            rospy.loginfo("Key 'g' pressed. Starting to trace the ArUco tag.")
            self.tracing_enabled = True
            if self.control_thread is None or not self.control_thread.is_alive():
                self.start_tracing()
        elif key == ord('b'):
            rospy.loginfo("Key 'b' pressed. Stopping tracing.")
            self.tracing_enabled = False
            self.stop_tracing()

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

    def update_current_target_pose(self):
        """
        Update the current target pose based on the last detected tag pose.
        """
        if self.last_tag_pose_base is None:
            rospy.logwarn("No tag pose available to move to.")
            return

        target_pose = Pose()
        target_pose.position = self.last_tag_pose_base.position
        target_pose.orientation = self.last_tag_pose_base.orientation

        # Adjust offsets to avoid collision with the tag
        safety_offset = 0.1  # Move 1 cm towards the tag (adjust as needed)
        target_pose.position.y += safety_offset

        # Set the end-effector orientation to point towards the tag
        # Here we maintain the current orientation
        try:
            # Get the current end-effector orientation
            transform_base_to_ee = self.tf_buffer.lookup_transform(
                self.robot_base_frame,
                self.end_effector_frame,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            target_pose.orientation = transform_base_to_ee.transform.rotation
        except Exception as e:
            rospy.logerr(f"Failed to get current end-effector orientation: {e}")
            return

        # Update the shared current target pose
        self.current_target_pose = target_pose

        # Publish the target pose for visualization
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.robot_base_frame
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose = target_pose
        self.target_pose_pub.publish(pose_stamped)

    def start_tracing(self):
        """
        Start the control thread for tracing the tag.
        """
        self.stop_thread.clear()
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()

    def stop_tracing(self):
        """
        Stop the control thread.
        """
        self.stop_thread.set()
        if self.control_thread is not None:
            self.control_thread.join()
            self.control_thread = None

    def control_loop(self):
        """
        Control loop to continuously send target poses to the robot.
        """
        rate = rospy.Rate(5)  # Control loop rate in Hz
        while not rospy.is_shutdown() and not self.stop_thread.is_set():
            if self.current_target_pose is not None:
                # Prepare position and orientation arrays
                position = [self.current_target_pose.position.x,
                            self.current_target_pose.position.y,
                            self.current_target_pose.position.z]
                orientation = [self.current_target_pose.orientation.x,
                               self.current_target_pose.orientation.y,
                               self.current_target_pose.orientation.z,
                               self.current_target_pose.orientation.w]

                # Send goal to the action server without waiting for the result
                goal = ArmPoseGoal()
                goal.pose.header.frame_id = self.robot_base_frame
                goal.pose.header.stamp = rospy.Time.now()
                goal.pose.pose.position.x = position[0]
                goal.pose.pose.position.y = position[1]
                goal.pose.pose.position.z = position[2]
                goal.pose.pose.orientation.x = orientation[0]
                goal.pose.pose.orientation.y = orientation[1]
                goal.pose.pose.orientation.z = orientation[2]
                goal.pose.pose.orientation.w = orientation[3]

                self.arm_pose_client.send_goal(goal)

            rate.sleep()

        # Optionally cancel any ongoing goals when stopping
        self.arm_pose_client.cancel_all_goals()

    def run(self):
        rospy.loginfo("Aruco Tracker is running. Press 'g' to start tracing the tag and 'b' to stop tracing.")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            self.stop_tracing()

if __name__ == '__main__':
    try:
        tracker = ArucoTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass
