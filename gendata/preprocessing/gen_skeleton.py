import numpy as np
import os
import h5py
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline as us


def b_spline(data, num_points, n_tail_coords):
    """
    Calculate a B-spline for given data to provide smoothly interpolated keypoints.

    Args:
        data: List of dictionaries containing 'keypoints' information.
        num_points: Integer, number of points to be interpolated on the spline.

    Returns:
        xy_coordinates: List containing evenly interpolated x and y coordinates.
    """

    # Create empty lists to store coordinate information
    x_even = []
    y_even = []
    xy_coordinates = []

    # Set the smoothing_factor; a larger value produces smoother curves
    smoothing_factor = 20

    # Extract keypoint information for each image and generate the corresponding plots
    for i in range(len(data)):
        keypoints = data[i]

        # Extract x and y coordinates
        x = np.array([kp[0] for kp in keypoints])
        y = np.array([kp[1] for kp in keypoints])

        # Find a new variable t, which is a function describing the path and relates to both x and y
        t = np.zeros(n_tail_coords)
        t[1:] = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
        t = np.cumsum(t)
        t /= t[-1]

        uq = np.linspace(int(np.min(x)), int(np.max(x)), 101)
        nt = np.linspace(0, 1, 100)

        # Calculate cubic spline
        spline_y_coords = us(t, y, k=3, s=smoothing_factor)(nt)
        spline_x_coords = us(t, x, k=3, s=smoothing_factor)(nt)

        # Obtain evenly-spaced spline indices
        spline_coords = np.array([spline_y_coords, spline_x_coords])

        # Evenly spaced points
        spline_nums = np.linspace(0, spline_coords.shape[1] - 1, num_points).astype(int)

        # Select evenly-spaced points along the spline
        spline_coords = spline_coords[:, spline_nums]
        y_new = spline_coords[0].tolist()
        x_new = spline_coords[1].tolist()

        x_even.append(x_new)
        y_even.append(y_new)

    xy_coordinates.append(x_even)
    xy_coordinates.append(y_even)

    return xy_coordinates


def load_and_process_data(data_path, num_points, tail_points):

    with h5py.File(data_path, 'r') as file:
        ds_x = np.array(file['ds_x'][...])
        ds_y = np.array(file['ds_y'][...])

    # x_head, x_tail = ds_x[:, :3], ds_x[:, 3:]
    # y_head, y_tail = ds_y[:, :3], ds_y[:, 3:]

    # x_tail = [[tail[0], tail[2], tail[1], tail[5], tail[4], tail[6], tail[3]] for tail in x_tail]
    # y_tail = [[tail[0], tail[2], tail[1], tail[5], tail[4], tail[6], tail[3]] for tail in y_tail]

    # merged_data = [list(zip(x_list, y_list)) for x_list, y_list in zip(x_tail, y_tail)]
    # merged_data = [list(zip(x_list, y_list)) for x_list, y_list in zip(ds_x, ds_y)]
    # interpolated_data = b_spline(merged_data, num_points, tail_points)
    #
    # ds_x = np.concatenate((x_head, interpolated_data[0]), axis=1)
    # ds_y = np.concatenate((y_head, interpolated_data[1]), axis=1)

    return [list(zip(x_list, y_list)) for x_list, y_list in zip(ds_x, ds_y)]


def create_ske_data(longest_bout, tail_data):
    """
    Repeat each bout in the keypoint sequences until it matches the length of the longest bout.

    :param longest_bout: An integer representing the length of the longest bout.
    :param tail_data: A list of numpy arrays representing the keypoint sequences.
    :return: A numpy array with repeated bouts.
    """
    repeated_sequences = []

    for bout in tail_data:
        bout = np.array(bout)
        current_length = bout.shape[0]

        # Calculate the number of times to repeat the bout
        repeat_factor = np.ceil(longest_bout / current_length).astype(int)

        # Repeat the bout and trim if necessary
        repeated_bout = np.tile(bout, (repeat_factor, 1, 1))[:longest_bout]

        repeated_sequences.append(repeated_bout)

    return np.array(repeated_sequences)


def extract_bouts(data, bout_starts, bout_ends):
    """
    Extracts bouts from the full sequence of keypoints.

    :param sequence: A numpy array of shape (32000, 16, 2) representing the full keypoint sequence.
    :param bout_starts: A list or array containing the start frames of each bout.
    :param bout_ends: A list or array containing the end frames of each bout.
    :return: A list of numpy arrays, each representing a bout.
    """
    bouts = []
    for start, end in zip(bout_starts, bout_ends):
        bout = data[start:end]  # Extract each bout as a NumPy array
        bouts.append(bout)
    return bouts


def convert_bouts_to_array(bouts):
    """
    Converts a list of bouts into a numpy array with padding.

    :param bouts: A list of numpy arrays, each representing a bout.
    :return: A numpy array of shape (550, N, 16, 2), where N is the length of the longest bout.
    """
    # Find the length of the longest bout
    max_length = max(bout.shape[0] for bout in bouts)

    # Initialize an array with zeros for padding
    padded_bouts = np.zeros((len(bouts), max_length, 16, 2))

    # Fill the array with bouts data
    for i, bout in enumerate(bouts):
        length = bout.shape[0]
        padded_bouts[i, :length, :, :] = bout

    return padded_bouts


def save_skeleton_files(interpolated_data, directory):
    """
    Saves each bout as a .skeleton file with a custom naming structure.

    :param interpolated_data: Numpy array of shape (550, 210, 16, 2) containing the interpolated bouts.
    :param directory: Directory where the .skeleton files will be saved.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    num_fish = 1  # Assuming all bouts are from the same fish
    num_bouts = interpolated_data.shape[0]
    num_frames = interpolated_data.shape[1]
    num_joints = interpolated_data.shape[2]

    skeleton_file_list = []  # List to store the file names

    for bout_index in range(num_bouts):
        file_name = f"fish{num_fish:04d}bout{bout_index + 1:04d}"
        skeleton_file_path = os.path.join(directory, file_name + ".skeleton")

        with open(skeleton_file_path, 'w') as file:
            # Write the number of frames
            file.write(f"{num_frames}\n")

            # Write the joint data for each frame
            for frame in range(num_frames):
                # Add num_fish and num_joints before each frame's data
                file.write(f"{num_fish}\n{num_joints}\n")

                for joint in range(num_joints):
                    x, y = interpolated_data[bout_index, frame, joint]
                    file.write(f"{x} {y}\n")

        # Add the file name (without extension) to the list
        skeleton_file_list.append(file_name)

    # Write the file names to a .txt file
    list_file_path = "../zebskes_210/statistics/skes_available_name.txt"
    with open(list_file_path, 'w') as list_file:
        for file_name in skeleton_file_list:
            list_file.write(file_name + "\n")


if __name__ == '__main__':
    data_path = '/Users/lanxinxu/Desktop/ES_10V_16x_Gcamp6s_mRubby_7dpf_42.h5'

    interp_points = 13
    tail_points = 9

    data = load_and_process_data(data_path, interp_points, tail_points)

    all_bout_starts = np.load('./bout_starts.npy')
    all_bout_ends = np.load('./bout_ends.npy')

    all_bout_length = all_bout_ends - all_bout_starts
    longest_bout = max(all_bout_length)

    extracted_data = extract_bouts(data, all_bout_starts, all_bout_ends)
    # bout_array = convert_bouts_to_array(extracted_data)
    interpolated_bout = create_ske_data(longest_bout, extracted_data)

    directory = "../zebskes_210/raw_data/skeleton_files"
    save_skeleton_files(interpolated_bout, directory)

    print('')

