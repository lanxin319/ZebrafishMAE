import os
import cv2
import math
import numpy as np
import re
import h5py
from scipy.interpolate import UnivariateSpline as us


def coordinates_to_angles(x, y):
    """
    Converts keypoint coordinates into corresponding angular information in radians.

    Args:
        x, y: Lists containing the x and y coordinates of the keypoints

    Returns:
        angle_info: List of angles in radians for each keypoint
    """

    angle_info = []

    for i in range(len(x)):
        n_points = len(x[i])
        angle_list = [0] * (n_points - 1)

        for z in range(1, n_points):
            # Rotate by 180 degrees and calculate the angle relative to the first point
            dx = x[i][0] - x[i][z]
            dy = y[i][0] - y[i][z]

            theta = math.atan2(dy, dx)

            # Check if previous angle exists and adjust based on its value
            if z > 0:
                diff = theta - angle_list[z-1]
                if diff > 2.8:  # difference larger than 180 degrees in radians
                    theta -= 2 * math.pi
                elif diff < -2.8:  # difference less than -180 degrees in radians
                    theta += 2 * math.pi

            # Store the angle
            angle_list[z - 1] = theta

        angle_info.append(angle_list)

    return angle_info


def yolo2voc(data):
    """
    Converts normalized keypoint information back to original coordinates from YOLO bounding box format.

    Args:
        data: List, containing coordinates of YOLO bounding box (center x, center y, width, height) and keypoint
        coordinates (x, y, visibility)

    Returns:
        keypoints: List of tuples, coordinates of all keypoints and their visibility status
    """

    # Actual dimensions of the image
    width_img = 220  # Width of the image
    height_img = 270  # Height of the image

    # Parsing keypoint coordinates and visibility
    keypoints_data = data[5:]
    keypoints = [(keypoints_data[i] * width_img, keypoints_data[i+1] * height_img, keypoints_data[i+2]) for i in
                 range(0, len(keypoints_data), 3)]

    return keypoints


def get_keypoints(folder_path):
    """
    Extracts keypoint information from all txt files in a folder and stores it.

    Args:
        folder_path: String, path to the folder

    Returns:
        all_data: List of tuples, information of all keypoints
    """

    # Initialize a list to hold data from all files
    all_data = []
    # List all files in the directory with .txt extension
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    file_list = sorted(file_list, key=lambda x: int(re.search(r'(\d+)\.txt$', x).group(1)))

    # Loop over each file
    for file_name in file_list:
        # Construct full file path
        file_path = os.path.join(folder_path, file_name)

        # Open and read the file
        with open(file_path, 'r') as file:
            # Read all lines in the file
            lines = file.readlines()[0]
            # Split the line into parts and convert each part to float, except the first one which should be an integer
            data = [float(x) if i else int(x) for i, x in enumerate(lines.strip().split())]

            # Assuming the file contains a list of keypoints, one per line
            keypoints = yolo2voc(data)
            all_data.append(keypoints)

    return all_data


def draw_keypoints(frame, keypoints, color=(0, 255, 0), radius=3, thickness=-1):
    """
    Draws keypoints on an image.

    Args:
        frame: The image on which to draw.
        keypoints: A list of keypoints, each keypoint represented as (x, y, confidence).
        color: The color to draw the keypoints, default is green.
        radius: The radius of the circles to draw for each keypoint.
        thickness: The thickness of the circles' outline. Default is -1, which means circles are filled.
    """

    for x, y, confidence in keypoints:
        if confidence > 0:  # 只有当置信度大于0时才绘制
            center = (int(x), int(y))
            cv2.circle(frame, center, radius, color, thickness)


def b_spline(data, num_points):
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

    # Number of points on the tail
    n_tail_coords = 7

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


if __name__ == '__main__':

    # Path to the folder containing keypoints data
    folder_path = '/Users/lanxinxu/Desktop/INTERN_2023/ES_10V_16x_Gcamp6s/ES_10V_16x_Gcamp6s-mRubby_7dpf_42'
    # Extract keypoints from the folder
    all_data = get_keypoints(folder_path)

    # Create an HDF5 file to store the data
    h5_file_path = '/Users/lanxinxu/Desktop/ES_10V_16x_Gcamp6s_mRubby_7dpf_42.h5'
    h5_file = h5py.File(h5_file_path, 'w')

    # Interpolate to a specified number of points for analysis
    num_points = 13  # The number of points to interpolate to
    interpolated_data = b_spline(all_data, num_points)

    print('data interpolated')

    # Separate out X and Y coordinates from the interpolated data
    ds_x = interpolated_data[0][1:]
    ds_y = interpolated_data[1][1:]

    # Convert keypoint coordinates to angular information
    angles = coordinates_to_angles(ds_x, ds_y)

    print('angle calculated')

    # Create datasets in the HDF5 file
    h5_file.create_dataset('ds_x', data=ds_x)
    h5_file.create_dataset('ds_y', data=ds_y)
    h5_file.create_dataset('ds_theta', data=angles)

    h5_file.close()

    print('Data has been stored as an h5 file')


