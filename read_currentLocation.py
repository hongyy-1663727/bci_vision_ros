#!/usr/bin/env python3

import rospy
import moveit_commander
from geometry_msgs.msg import Pose

class MoveRobot():
    def __init__(self):
        # Initialize moveit_commander with the appropriate joint state topic
        joint_state_topic = ['joint_states:=/j2n6s300_driver/out/joint_state']
        moveit_commander.roscpp_initialize(joint_state_topic)

        # Initialize robot and arm group
        self.robot = moveit_commander.RobotCommander()
        self.arm_group = moveit_commander.MoveGroupCommander("arm")

        # Set arm's speed and acceleration scaling factors (optional)
        self.arm_group.set_max_acceleration_scaling_factor(0.2)
        self.arm_group.set_max_velocity_scaling_factor(0.2)

    def main_loop(self):
        """
        Main loop that retrieves and prints the current end-effector position.
        """
        # Retrieve the current pose of the end-effector
        current_pose = self.arm_group.get_current_pose().pose

        # Extract x, y, z positions
        current_x = current_pose.position.x
        current_y = current_pose.position.y
        current_z = current_pose.position.z

        # Print the current position
        rospy.loginfo(f"Current End-Effector Position -> x: {current_x:.3f}, y: {current_y:.3f}, z: {current_z:.3f}")

def main():
    rospy.init_node('grasp_demo', anonymous=True)
    rospy.loginfo('Start Grasp Demo')
    moverobot = MoveRobot()

    rate = rospy.Rate(1)  # 1 Hz update rate

    while not rospy.is_shutdown():
        moverobot.main_loop()
        rate.sleep()  # Maintain the loop rate

if __name__ == "__main__":
    main()
