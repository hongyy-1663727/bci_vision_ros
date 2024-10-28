#!/usr/bin/env python3

import rospy
import moveit_commander
from geometry_msgs.msg import Pose

class MoveRobot():
    def __init__(self):
        joint_state_topic = ['joint_states:=/j2n6s300_driver/out/joint_state']
        moveit_commander.roscpp_initialize(joint_state_topic)

        # Initialize robot and arm group
        self.robot = moveit_commander.RobotCommander()
        self.arm_group = moveit_commander.MoveGroupCommander("arm")

        # Set arm's speed and acceleration
        self.arm_group.set_max_acceleration_scaling_factor(0.2)
        self.arm_group.set_max_velocity_scaling_factor(0.2)

    def main_loop(self):
        """
        Main loop with a set_pose_target method to move the robot.
        """
        target_pose = Pose()
        target_pose.position.x = 0.175
        target_pose.position.y = -0.593
        target_pose.position.z = 0.21
        target_pose.orientation.x = 0.0
        target_pose.orientation.y = 0.0
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.0

        # Set the target pose and execute
        self.arm_group.set_pose_target(target_pose)
        self.arm_group.go(wait=True)

def main():
    rospy.init_node('grasp_demo', anonymous=True)
    rospy.loginfo('Start Grasp Demo')
    moverobot = MoveRobot()

    while not rospy.is_shutdown():
        moverobot.main_loop()

if __name__ == "__main__":
    main()
