import matplotlib.pyplot as plt
import numpy as np
import json
import h5py
from utils import b_spline


def visualize_skeleton(frame_data, connections):
    plt.figure()
    for point in frame_data:
        plt.scatter(*point, c='blue')
    for i, j in connections:
        plt.plot([frame_data[i][0], frame_data[j][0]], [frame_data[i][1], frame_data[j][1]], c='red')
    plt.title("Skeleton Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def create_skeleton_data(data, connections):
    return [{'joints': frame, 'connections': connections} for frame in data]


def load_and_process_data(data_path, num_points):

    with h5py.File(data_path, 'r') as file:
        ds_x = np.array(file['ds_x'][...])
        ds_y = np.array(file['ds_y'][...])

    x_head, x_tail = ds_x[:, :3], ds_x[:, 3:]
    y_head, y_tail = ds_y[:, :3], ds_y[:, 3:]

    x_tail = [[tail[0], tail[2], tail[1], tail[5], tail[4], tail[6], tail[3]] for tail in x_tail]
    y_tail = [[tail[0], tail[2], tail[1], tail[5], tail[4], tail[6], tail[3]] for tail in y_tail]

    merged_data = [list(zip(x_list, y_list)) for x_list, y_list in zip(x_tail, y_tail)]
    interpolated_data = b_spline(merged_data, num_points)

    ds_x = np.concatenate((x_head, interpolated_data[0]), axis=1)
    ds_y = np.concatenate((y_head, interpolated_data[1]), axis=1)

    return [list(zip(x_list, y_list)) for x_list, y_list in zip(ds_x, ds_y)]


if __name__ == '__main__':
    data_path = '/Users/lanxinxu/Desktop/UCL_Year_3/Epileptic_zebrafish_project/ZebraPoseMAE/gendata/preprocessing/test_data.h5'
    num_points = 13
    connections = [(0, 1), (0, 2), (1, 3), (2, 3)] + [(i, i+1) for i in range(3, 15)]

    data = load_and_process_data(data_path, num_points)

    visualize_skeleton(data[70], connections)

    skeleton_data = create_skeleton_data(data, connections)

    with open('/Users/lanxinxu/Desktop/skeleton_data.json', 'w') as file:
        json.dump(skeleton_data, file, indent=4)


















