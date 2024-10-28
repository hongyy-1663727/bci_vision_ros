
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection

def read_log_file(log_file):
    timestamps = []
    ee_positions = []
    tag_positions = []

    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp']))
            ee_positions.append([
                float(row['ee_pos_x']),
                float(row['ee_pos_y']),
                float(row['ee_pos_z'])
            ])
            tag_positions.append([
                float(row['tag_pos_x']),
                float(row['tag_pos_y']),
                float(row['tag_pos_z'])
            ])

    return np.array(timestamps), np.array(ee_positions), np.array(tag_positions)

def animate_robot_motion(timestamps, ee_positions, tag_positions, sphere_radius):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Robot End-Effector and ArUco Tag Positions Over Time')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # Setting the axes properties
    max_range = np.array([ee_positions[:,0].max()-ee_positions[:,0].min(),
                          ee_positions[:,1].max()-ee_positions[:,1].min(),
                          ee_positions[:,2].max()-ee_positions[:,2].min()]).max() / 2.0

    mid_x = (ee_positions[:,0].max()+ee_positions[:,0].min()) * 0.5
    mid_y = (ee_positions[:,1].max()+ee_positions[:,1].min()) * 0.5
    mid_z = (ee_positions[:,2].max()+ee_positions[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Initialize lines and points
    ee_line, = ax.plot([], [], [], 'b-', label='End-Effector Path')
    ee_point, = ax.plot([], [], [], 'bo', label='End-Effector')
    tag_line, = ax.plot([], [], [], 'r--', label='ArUco Tag Path')
    tag_point, = ax.plot([], [], [], 'ro', label='ArUco Tag')

    # Create sphere data
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    sphere_wireframe = []  # Mutable container to hold the sphere wireframe

    # Combine positions for drawing paths
    ee_path = []
    tag_path = []

    def init():
        ee_line.set_data([], [])
        ee_line.set_3d_properties([])
        ee_point.set_data([], [])
        ee_point.set_3d_properties([])

        tag_line.set_data([], [])
        tag_line.set_3d_properties([])
        tag_point.set_data([], [])
        tag_point.set_3d_properties([])

        # Initialize the sphere at the first tag position
        sphere_center = tag_positions[0]
        sphere_x = sphere_radius * np.cos(u) * np.sin(v) + sphere_center[0]
        sphere_y = sphere_radius * np.sin(u) * np.sin(v) + sphere_center[1]
        sphere_z = sphere_radius * np.cos(v) + sphere_center[2]
        wf = ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="green", alpha=0.3, label='Virtual Sphere')
        sphere_wireframe.append(wf)

        return ee_line, ee_point, tag_line, tag_point, wf

    def update(frame):
        # Update paths
        ee_path.append(ee_positions[frame])
        tag_path.append(tag_positions[frame])

        # Update end-effector line and point
        ee_line.set_data(np.array(ee_path)[:,0], np.array(ee_path)[:,1])
        ee_line.set_3d_properties(np.array(ee_path)[:,2])
        ee_point.set_data(ee_positions[frame,0], ee_positions[frame,1])
        ee_point.set_3d_properties(ee_positions[frame,2])

        # Update tag line and point
        tag_line.set_data(np.array(tag_path)[:,0], np.array(tag_path)[:,1])
        tag_line.set_3d_properties(np.array(tag_path)[:,2])
        tag_point.set_data(tag_positions[frame,0], tag_positions[frame,1])
        tag_point.set_3d_properties(tag_positions[frame,2])

        # Update sphere position
        sphere_center = tag_positions[frame]
        sphere_x = sphere_radius * np.cos(u) * np.sin(v) + sphere_center[0]
        sphere_y = sphere_radius * np.sin(u) * np.sin(v) + sphere_center[1]
        sphere_z = sphere_radius * np.cos(v) + sphere_center[2]

        # Remove previous sphere wireframe
        sphere_wireframe[0].remove()

        # Plot sphere at new position
        wf = ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="green", alpha=0.3, label='_nolegend_')
        sphere_wireframe[0] = wf

        return ee_line, ee_point, tag_line, tag_point, wf

    ani = FuncAnimation(fig, update, frames=range(len(timestamps)),
                        init_func=init, blit=False, interval=50)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    log_file = 'pose_log.csv'  # Make sure this matches your log file's name
    timestamps, ee_positions, tag_positions = read_log_file(log_file)
    sphere_radius = 0.1  # Set the sphere radius to the same value as in your main program
    animate_robot_motion(timestamps, ee_positions, tag_positions, sphere_radius)
