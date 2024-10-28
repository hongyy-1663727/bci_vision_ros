#!/usr/bin/env python3
import rospy
import threading
import sys
import tf2_ros
import moveit_commander
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from pynput import keyboard
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2

class ArucoMover:
    def __init__(self):
        # Initialize MoveIt
        joint_state_topic = ['joint_states:=/j2n6s300_driver/out/joint_state']
        moveit_commander.roscpp_initialize(joint_state_topic)
        self.robot = moveit_commander.RobotCommander()
        self.arm_group = moveit_commander.MoveGroupCommander("arm")
        self.arm_group.set_num_planning_attempts(10)
        self.arm_group.set_planning_time(5)

        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # CvBridge for image conversion
        self.bridge = CvBridge()

        # Latest marker pose
        self.marker_pose = None
        self.marker_lock = threading.Lock()

        # Subscribe to the ArUco tracker pose
        rospy.Subscriber("/aruco_tracker/pose", PoseStamped, self.aruco_callback)

        # Subscribe to the camera image
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        # Keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        rospy.loginfo("ArucoMover initialized and listening for 'g' key press...")

    def aruco_callback(self, msg):
        with self.marker_lock:
            self.marker_pose = msg
            #rospy.loginfo_throttle(5, "Received marker pose: %s", msg)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Overlay marker information if available
            with self.marker_lock:
                if self.marker_pose:
                    # Display marker position information
                    position = self.marker_pose.pose.position
                    text = f"Marker Pos: x={position.x:.2f}, y={position.y:.2f}, z={position.z:.2f}"
                    cv2.putText(cv_image, text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the image
            cv2.imshow("Camera Stream", cv_image)
            cv2.waitKey(1)  # Needed to refresh the image window

            rospy.loginfo_throttle(10, "Displaying camera stream.")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
        except Exception as e:
            rospy.logerr("Unexpected error in image_callback: {0}".format(e))

    def on_press(self, key):
        try:
            if key.char == 'g':
                rospy.loginfo("'g' key pressed. Attempting to move to marker.")
                self.move_to_marker()
        except AttributeError:
            # Special keys (like ctrl, alt, etc.) are ignored
            pass

    def move_to_marker(self):
        with self.marker_lock:
            if self.marker_pose is None:
                rospy.logwarn("No marker pose received yet.")
                return

            target_pose = Pose()
            #target_pose.header = self.marker_pose.header
            target_pose.position.x = self.marker_pose.pose.position.x
            target_pose.position.y = self.marker_pose.pose.position.y
            target_pose.position.z = self.marker_pose.pose.position.z
            target_pose.orientation.x = 0.0
            target_pose.orientation.y = 0.0
            target_pose.orientation.z = 0.0
            target_pose.orientation.w = 0.0

        # Set the target pose
        self.arm_group.set_pose_target(target_pose)
        rospy.loginfo("Set target pose for MoveIt.")

        # Plan and execute
        plan = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        if plan:
            rospy.loginfo("Move successful.")
        else:
            rospy.logwarn("Move failed or was not executed.")

    def shutdown(self):
        self.listener.stop()
        roscpp_shutdown()
        cv2.destroyAllWindows()
        rospy.loginfo("ArucoMover shutdown.")

def main():
    rospy.init_node('aruco_mover', anonymous=True)
    mover = ArucoMover()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down ArucoMover node.")
    finally:
        mover.shutdown()

if __name__ == '__main__':
    main()
