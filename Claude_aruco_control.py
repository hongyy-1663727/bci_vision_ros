#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander, RobotCommander
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion

class ArucoRobotController:
    def __init__(self):
        rospy.init_node('aruco_robot_controller')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize Robot Commander for getting robot state
        self.robot = RobotCommander()
        
        # Initialize MoveIt
        self.move_group = MoveGroupCommander("arm")
        self.move_group.set_planning_time(5)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.set_max_velocity_scaling_factor(0.1)
        self.move_group.set_max_acceleration_scaling_factor(0.1)
        
        # Get the end effector link name
        self.end_effector_link = self.move_group.get_end_effector_link()
        rospy.loginfo(f"End effector link: {self.end_effector_link}")
        
        # TF2 Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribe to camera feed
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', 
                                        Image, 
                                        self.image_callback)
        
        # Subscribe to camera info
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info',
                                              CameraInfo,
                                              self.camera_info_callback)
        
        # Subscribe to ArUco pose
        self.aruco_sub = rospy.Subscriber('/aruco_single/pose',
                                        PoseStamped,
                                        self.aruco_callback)
        
        # Store latest poses and status
        self.latest_aruco_pose = None
        self.aruco_base_pose = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.last_aruco_callback_time = None
        self.debug_info = ""
        
        # ArUco dictionary
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Image window name
        self.window_name = "Camera Feed with ArUco Detection"
        cv2.namedWindow(self.window_name)
        
        rospy.loginfo("ArUco Robot Controller initialized")
        
        # Start debug timer
        rospy.Timer(rospy.Duration(1.0), self.debug_timer_callback)

    def debug_timer_callback(self, event):
        # Check subscription status
        try:
            aruco_subscribers = rospy.get_published_topics('/aruco_single/pose')
            camera_subscribers = rospy.get_published_topics('/camera/color/image_raw')
            
            self.debug_info = "Debug Info:\n"
            self.debug_info += f"ArUco topic active: {len(aruco_subscribers) > 0}\n"
            self.debug_info += f"Camera topic active: {len(camera_subscribers) > 0}\n"
            
            if self.last_aruco_callback_time:
                time_since_last = rospy.Time.now() - self.last_aruco_callback_time
                self.debug_info += f"Time since last ArUco pose: {time_since_last.to_sec():.1f}s\n"
            else:
                self.debug_info += "No ArUco pose received yet\n"
                
        except Exception as e:
            self.debug_info += f"Error getting debug info: {str(e)}\n"
        
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            rospy.loginfo("Camera parameters received")
    
    def get_end_effector_pose(self):
        try:
            current_pose = self.move_group.get_current_pose().pose
            return current_pose
        except Exception as e:
            rospy.logerr(f"Error getting end effector pose: {e}")
            return None

    def transform_to_base_frame(self, pose_msg):
        try:
            # Transform from camera frame to robot base frame
            transform = self.tf_buffer.lookup_transform('root', 
                                                      pose_msg.header.frame_id,
                                                      rospy.Time(0),
                                                      rospy.Duration(1.0))
            
            pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_msg, transform)
            return pose_transformed
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF2 Error: {e}")
            return None

    def display_position_info(self, image):
        # Initialize y_offset for text positioning
        y_offset = 30
        line_height = 30

        # Display header
        cv2.putText(image, "Positions in Robot Base Frame:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height

        # Display time since last ArUco pose
        if self.last_aruco_callback_time:
            time_since_last = (rospy.Time.now() - self.last_aruco_callback_time).to_sec()
            cv2.putText(image, f"Last ArUco update: {time_since_last:.1f}s ago", (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(image, "Waiting for first ArUco pose...", (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y_offset += line_height

        # Display ArUco position if available
        if self.aruco_base_pose is not None:
            pos = self.aruco_base_pose.pose.position
            text = f"ArUco: X: {pos.x:.3f}, Y: {pos.y:.3f}, Z: {pos.z:.3f}"
            cv2.putText(image, text, (10, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image, "ArUco: Not transformed to base frame", (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += line_height

        # Display end effector position
        ee_pose = self.get_end_effector_pose()
        if ee_pose is not None:
            text = f"End Effector: X: {ee_pose.position.x:.3f}, Y: {ee_pose.position.y:.3f}, Z: {ee_pose.position.z:.3f}"
            cv2.putText(image, text, (10, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(image, "End Effector: Position unknown", (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y_offset += line_height

        # Display debug info
        debug_lines = self.debug_info.split('\n')
        for line in debug_lines:
            if line:
                cv2.putText(image, line, (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_offset += 25

        # Display movement instructions
        if self.aruco_base_pose is not None:
            cv2.putText(image, "Press 'g' to move robot to marker", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Waiting for ArUco pose transform...", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Detect ArUco markers
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            # Draw detected markers
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                cv2.putText(cv_image, f"Visual ArUco detection: ID {ids[0][0]}", (10, cv_image.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display position information
            self.display_position_info(cv_image)
            
            # Display image
            cv2.imshow(self.window_name, cv_image)
            key = cv2.waitKey(1) & 0xFF
            
            # Check for 'g' key press
            if key == ord('g') and self.aruco_base_pose is not None:
                rospy.loginfo("Moving to ArUco marker position...")
                self.move_to_aruco()
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def aruco_callback(self, msg):
        self.last_aruco_callback_time = rospy.Time.now()
        self.latest_aruco_pose = msg
        rospy.loginfo_throttle(1, f"Received ArUco pose in frame: {msg.header.frame_id}")
        
        # Transform ArUco pose to base frame
        self.aruco_base_pose = self.transform_to_base_frame(msg)
        if self.aruco_base_pose:
            rospy.loginfo_throttle(1, "Successfully transformed ArUco pose to base frame")
        else:
            rospy.logwarn_throttle(1, "Failed to transform ArUco pose to base frame")
    
    def move_to_aruco(self):
        try:
            if self.aruco_base_pose is None:
                rospy.logwarn("No ArUco pose available in base frame")
                return
            
            # Set the target pose for the end-effector
            target_pose = self.aruco_base_pose.pose
            
            # Add some offset in Z to avoid collision (adjust as needed)
            target_pose.position.z += 0.1
            
            # Plan and execute movement
            rospy.loginfo("Planning movement to target pose...")
            self.move_group.set_pose_target(target_pose)
            success = self.move_group.go(wait=True)
            
            if success:
                rospy.loginfo("Successfully moved to ArUco marker!")
            else:
                rospy.logwarn("Failed to plan/execute movement to ArUco marker")
            
            # Clear targets
            self.move_group.clear_pose_targets()
            
        except Exception as e:
            rospy.logerr(f"Error moving to ArUco: {e}")
    
    def cleanup(self):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        controller = ArucoRobotController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        controller.cleanup()