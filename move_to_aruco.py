#!/usr/bin/env python3

import rospy
import sys
import copy
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, roscpp_shutdown
from moveit_msgs.msg import MoveItErrorCodes

class MoveToAruco:
    def __init__(self):
        # Initialize ROS node and MoveIt
        roscpp_initialize(sys.argv)
        self.robot = RobotCommander()
        self.group = MoveGroupCommander("arm")  # Ensure "arm" is your correct MoveIt group name

        # Set planner parameters
        self.group.set_planner_id("RRTConnectkConfigDefault")
        self.group.set_num_planning_attempts(10)
        self.group.set_planning_time(30)  # Increased planning time
        self.group.set_max_velocity_scaling_factor(0.2)  # Reduced speed
        self.group.set_max_acceleration_scaling_factor(0.2)  # Reduced acceleration

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribe to ArUco marker pose
        self.aruco_sub = rospy.Subscriber("/aruco_tracker/pose", PoseStamped, self.aruco_callback)

        rospy.loginfo("MoveToAruco node initialized and waiting for ArUco markers...")

    def aruco_callback(self, msg):
        rospy.loginfo("Received ArUco marker pose.")

        try:
            # Define target frame (robot's base frame)
            target_frame = "j2n6s300_link_base"  # Replace with your robot's actual base frame if different

            # Check available transform
            if not self.tf_buffer.can_transform(target_frame, msg.header.frame_id, rospy.Time(), rospy.Duration(4.0)):
                rospy.logwarn("Transform from %s to %s not available.", msg.header.frame_id, target_frame)
                return
            rospy.loginfo("Transform available from %s to %s.", msg.header.frame_id, target_frame)

            # Transform the pose to the target frame
            transformed_pose = self.tf_buffer.transform(msg, target_frame, rospy.Duration(1.0))
            rospy.loginfo("Transformed pose to base frame.")

            # Optional: Adjust the target pose if needed (e.g., add an offset)
            target_pose = copy.deepcopy(transformed_pose)
            target_pose.pose.orientation.w = 1.0  # Neutral orientation

            # Log the target pose for debugging
            rospy.loginfo("Target Pose: Position -> x: %.3f, y: %.3f, z: %.3f", 
                          target_pose.pose.position.x, 
                          target_pose.pose.position.y, 
                          target_pose.pose.position.z)

            # Set the target pose
            self.group.set_pose_target(target_pose)

            # Plan to the new pose
            plan_result = self.group.plan()

            # Debug: Log the type and contents of plan_result
            rospy.logdebug("Plan result type: %s", type(plan_result))
            rospy.logdebug("Plan result content: %s", plan_result)

            # Handle different possible return types based on the number of returned values
            if isinstance(plan_result, tuple):
                num_returns = len(plan_result)
                if num_returns == 4:
                    plan, planning_time, error_code, _ = plan_result
                elif num_returns == 3:
                    plan, planning_time, error_code = plan_result
                elif num_returns == 2:
                    plan, planning_time = plan_result
                else:
                    rospy.logerr("Unexpected number of return values from plan(): %d", num_returns)
                    return
            else:
                plan = plan_result  # Assuming it's a RobotTrajectory

            # Check if the plan is valid
            if isinstance(plan, MoveItErrorCodes):
                rospy.logwarn("MoveIt! returned an error code: %s", plan)
                return

            # Depending on the plan's type, adjust the check
            if hasattr(plan, 'joint_trajectory') and len(plan.joint_trajectory.points) > 0:
                rospy.loginfo("Plan found. Executing...")
                exec_status = self.group.go(wait=True)
                self.group.stop()
                self.group.clear_pose_targets()

                if exec_status:
                    rospy.loginfo("Move executed successfully.")
                else:
                    rospy.logwarn("Move execution failed.")
            else:
                rospy.logwarn("No valid plan found.")
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
            rospy.logerr("TF transformation failed: %s", e)
        except Exception as e:
            rospy.logerr("An unexpected error occurred: %s", e)

    def shutdown(self):
        roscpp_shutdown()

def main():
    rospy.init_node('move_to_aruco_node', anonymous=True)
    mover = MoveToAruco()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down MoveToAruco node.")
    finally:
        mover.shutdown()

if __name__ == '__main__':
    main()
