#!/usr/bin/env python3
import rospy
import cv2
import numpy as np

from geometry_msgs.msg import Pose
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
import sys

class EndEffectorMover:
    def __init__(self):
        # Initialize MoveIt! commander and rospy node
        roscpp_initialize(sys.argv)
        rospy.init_node('end_effector_mover_node', anonymous=True)

        # Initialize MoveGroupCommander for the robot's arm
        self.move_group = MoveGroupCommander('arm')  # Replace 'arm' with your Move Group name

        # Allow some time for MoveIt! to initialize
        rospy.sleep(2)

        # Get the current pose of the end-effector
        self.current_pose = self.move_group.get_current_pose().pose
        rospy.loginfo(f"Initial End Effector Pose: {self.current_pose}")

        # Create an OpenCV window
        cv2.namedWindow("End Effector Mover", cv2.WINDOW_NORMAL)
        cv2.imshow("End Effector Mover", np.zeros((100, 300), dtype=np.uint8))  # Display a blank image

    def move_end_effector(self):
        # Update the current pose
        self.current_pose = self.move_group.get_current_pose().pose

        # Increment the X position by 0.01 meters
        target_pose = Pose()
        target_pose.position.x = self.current_pose.position.x + 0.01
        target_pose.position.y = self.current_pose.position.y
        target_pose.position.z = self.current_pose.position.z

        # Maintain current orientation
        target_pose.orientation = self.current_pose.orientation

        rospy.loginfo(f"Moving End Effector to: {target_pose}")

        # Set the target pose
        self.move_group.set_pose_target(target_pose)

        # Plan and execute the motion
        plan = self.move_group.go(wait=True)

        # Stop any residual movement
        self.move_group.stop()

        # Clear targets after planning
        self.move_group.clear_pose_targets()

        if plan:
            rospy.loginfo("End effector moved successfully.")
        else:
            rospy.logwarn("Failed to move end effector.")

    def run(self):
        rospy.loginfo("End Effector Mover is running. Press 'g' to move the end-effector +0.01m along X-axis. Press 'q' to quit.")

        while not rospy.is_shutdown():
            # Wait for a key press for 100ms
            key = cv2.waitKey(100) & 0xFF

            if key == ord('g'):
                self.move_end_effector()
            elif key == ord('q'):
                rospy.loginfo("Quitting End Effector Mover.")
                break

        # Cleanup
        cv2.destroyAllWindows()
        roscpp_shutdown()

if __name__ == '__main__':
    try:
        mover = EndEffectorMover()
        mover.run()
    except rospy.ROSInterruptException:
        pass
