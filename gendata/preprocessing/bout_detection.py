import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.ndimage import maximum_filter1d, minimum_filter1d


def boxcarf(bclength):
    return np.ones(bclength) / bclength


class Zebrafish:

    def __init__(self, ds_x, ds_y, ds_theta):
        self.ds_x = ds_x
        self.ds_y = ds_y
        self.ds_theta = ds_theta

        # Find the min length of data
        min_length = min(len(self.ds_x), len(self.ds_y), len(self.ds_theta))

        # Make sure all length the same
        self.ds_x = self.ds_x[:, :min_length]
        self.ds_y = self.ds_y[:, :min_length]
        self.ds_theta = self.ds_theta[:, :min_length]

    def interpolate_tail_data(self):
        """
        Interpolates tail data both spatially and temporally.

        Returns:
            ds_theta_700hz (array): The interpolated tail angles at 700Hz.
            ds_theta_200hz (array): The interpolated tail angles at 200Hz.
            x_time (array): Original time points.
            xi_time (array): Interpolated time points.
        """
        # Normalized angles
        ds_theta_norm = np.zeros_like(self.ds_theta)
        for h in range(self.ds_theta.shape[0]):
            ds_theta_norm[h, :] = ds_theta[h, :]

        xi = np.arange(0, 310 * 8 + 1, 310)
        x_perc = np.arange(0, 80 * 36 + 1, 80)

        ds_theta_norm_interp = np.zeros((ds_theta_norm.shape[0], len(xi)))
        for i in range(ds_theta_norm.shape[0]):
            interp_func = interp1d(x_perc, ds_theta_norm[i, :], kind='cubic', fill_value='extrapolate')
            ds_theta_norm_interp[i, :] = interp_func(xi)

        # Interpolate data into 700Hz
        x_time = np.arange(1, len(ds_x) + 1)
        xi_time = np.arange(1, len(ds_x) + 1, 200 / 700)
        interp_func_theta = interp1d(x_time, ds_theta, kind='cubic', axis=0, fill_value='extrapolate')

        ds_theta_700hz = interp_func_theta(xi_time)
        ds_theta_200hz = ds_theta_norm_interp

        return ds_theta_700hz, ds_theta_200hz, x_time, xi_time

    def tail_smoother(self, interpolated_data, bc_size):
        """
        Smooths the data for tail movement detection.

        Args:
            bc_size (int): The window size for the moving average filter.

        Returns:
            numpy.ndarray: The smoothed tail curvature measure. (Should be 1-D)
        """

        # Apply a moving average filter to each element, except the first and last
        for n in range(1, interpolated_data.shape[1] - 1):
            interpolated_data[:, n] = np.mean(interpolated_data[:, (n - 1):(n + 2)], axis=1)

        segment_angles = np.diff(interpolated_data, axis=0)
        # Append a column of zeros to the last column to store curvature measure
        segment_angles = np.hstack([segment_angles, np.zeros((np.array(segment_angles).shape[0], 1))])

        # Create a zero array with the same shape as segment_angles
        filtered_angles = np.zeros_like(segment_angles)

        # Apply a convolution filter to each column
        for n in range(segment_angles.shape[1]):
            # Create a convolution kernel and perform the convolution operation
            kernel = np.ones(bc_size) / bc_size
            filtered_angles[:, n] = convolve(segment_angles[:, n], kernel, mode='same')

        # Calculate the cumulative sum of the absolute values of the filtered angles
        cum_sum_angles = np.cumsum(abs(filtered_angles), axis=1)

        # Calculate the cumulative sum of the absolute values of cum_sum_angles
        super_cum_sum_angles = np.cumsum(abs(cum_sum_angles), axis=1)

        # Convolve the last column of super_cum_sum_angles with a boxcar filter
        tail_curve_measure = convolve(super_cum_sum_angles[:, -1], boxcarf(bc_size), mode='same')

        # # Apply max and min filters
        max_filtered = maximum_filter1d(tail_curve_measure[:], size=20)  # 700帧对20
        min_filtered = minimum_filter1d(tail_curve_measure[:], size=400)  # 700帧对400

        # Calculate the smoothed tail curvature measure
        smoother_tail_curve_measure = max_filtered - min_filtered
        smoother_tail_curve_measure = tail_curve_measure

        # # Correct any size discrepancy
        # if len(smoother_tail_curve_measure) != len(cum_sum_angles):
        #     smoother_tail_curve_measure = smoother_tail_curve_measure[:-1]

        # 将 smootherTailCurveMeasure 中的 NaN 和 Inf 值替换为 0
        smoother_tail_curve_measure[np.isnan(smoother_tail_curve_measure)] = 0
        smoother_tail_curve_measure[np.isinf(smoother_tail_curve_measure)] = 0

        return smoother_tail_curve_measure

    def bout_detector(self, smoother_tail_curve_measure, min_interval=8, min_length=30):
        """
        Detects motion sequences (bouts) in the smoother_tail_curve_measure data.

        Args:
            smoother_tail_curve_measure (array): An array containing the smoother curve measurements.
            min_interval (int, optional): Minimum interval between bouts. Defaults to 10.
            min_length (int, optional): Minimum length of a bout. Defaults to 30.

        Returns:
            tuple: A tuple containing the start and end indices of detected bouts, and the detection threshold.
        """
        threshold = 0.2  # Can be changed

        # Remove outliers due to tracking mistakes
        smoother_tail_curve_measure_clean = smoother_tail_curve_measure.copy()
        ind_outlyer = np.where(smoother_tail_curve_measure > 20)[0]  # discard data > 20 (outliers)
        smoother_tail_curve_measure_clean[ind_outlyer] = 0

        # Find bouts by threshold
        all_bouts = np.where(
            np.diff(smoother_tail_curve_measure_clean[1:len(smoother_tail_curve_measure)] > threshold))[0]

        # Adjust for starting in the middle of a bout
        if smoother_tail_curve_measure[0] > threshold:
            all_bout_starts = all_bouts[1::2]
            all_bout_ends = all_bouts[2::2]
        else:
            all_bout_starts = all_bouts[0::2]
            all_bout_ends = all_bouts[1::2]

        # Ensure that starts and ends are of the same length
        all_bout_starts = all_bout_starts[:len(all_bout_ends)]

        # Calculate the inter-bout lengths
        all_inter_bout_lengths = all_bout_starts[1:] - all_bout_ends[:-1]

        # Merge bouts that are too close
        short_intervals = np.where(all_inter_bout_lengths < min_interval)[0]
        while short_intervals.size > 0:
            short_interval = short_intervals[0]

            if short_interval < len(all_bout_starts) - 2:
                all_bout_starts = np.delete(all_bout_starts, short_interval + 1)
            else:
                all_bout_starts = all_bout_starts[:short_interval + 1]

            if short_interval > 0:
                all_bout_ends = np.delete(all_bout_ends, short_interval)
            else:
                all_bout_ends = all_bout_ends[1:]

            # Update bout starts and ends
            all_inter_bout_lengths = all_bout_starts[1:] - all_bout_ends[:-1]
            short_intervals = np.where(all_inter_bout_lengths < min_interval)[0]

        # Delete bouts that are too short
        bout_durations = all_bout_ends - all_bout_starts
        valid_bouts = bout_durations > min_length
        all_bout_starts = all_bout_starts[valid_bouts]
        all_bout_ends = all_bout_ends[valid_bouts]

        if len(all_bout_starts) > 0:
            # Select curvature block for current stimulus type
            curvature_block = smoother_tail_curve_measure

            # Identify good bouts based on curvature
            good_starts = [n for n in range(len(all_bout_starts)) if
                           max(curvature_block[all_bout_starts[n]:all_bout_ends[n]]) >= 0.4]  #0.45

            # Filter out bouts with small curvature
            all_bout_starts = all_bout_starts[good_starts]
            all_bout_ends = all_bout_ends[good_starts]
        else:
            print('The length of all_bout_starts is 0!')

        return all_bout_starts, all_bout_ends, threshold


def align_fish(data, p1=0, p2=3):
    """
    Aligns fish by rotating points around the 3rd point in each frame based on the
    body angle between points p1 and p2, making the coordinate system egocentric
    with respect to the 6th point.

    Args:
        data (list of lists): Data where each element is a list containing keypoints
                              in the format [x, y, visibility].
        p1 (int): Index for the first point used to determine the body angle.
        p2 (int): Index for the second point used to determine the body angle.

    Returns:
        list: The rotated and translated data.
    """

    for i in range(len(data)):
        # Get the coordinates of the 3rd point
        center_x, center_y = data[i][p2][:2]

        # Calculate the body angle in radians
        dx = data[i][p2][0] - data[i][p1][0]
        dy = data[i][p2][1] - data[i][p1][1]
        body_angle = math.atan2(dy, dx)

        # Calculate the angle needed to align the body vertically with head up
        target_angle = math.pi / 2  # 90 degrees in radians
        rotation_angle = target_angle - body_angle + math.pi  # Add 180 degrees (pi radians)

        # Precompute cosine and sine for rotation
        cos_angle = math.cos(rotation_angle)
        sin_angle = math.sin(rotation_angle)

        for j in range(len(data[i])):

            x, y, visibility = data[i][j]

            # Translate to make the 3rd point the origin
            x -= center_x
            y -= center_y

            # Apply the rotation matrix to the point
            x_new = x * cos_angle - y * sin_angle
            y_new = x * sin_angle + y * cos_angle

            # Update the point in the original data
            data[i][j] = [x_new, y_new, visibility]

    return data


def convert_frames_to_200(all_bout_starts, all_bout_ends):
    """
    Convert frame indices from 700 frames format to 200 frames format.

    Args:
        all_bout_starts (array): An array of start indices of bouts in 700 frames format.
        all_bout_ends (array): An array of end indices of bouts in 700 frames format.

    Returns:
        tuple: Two arrays containing converted start and end indices in 200 frames format.
    """

    # Conversion ratio
    conversion_ratio = 200 / 700

    # Convert all_bout_starts and all_bout_ends
    converted_starts = np.round(all_bout_starts * conversion_ratio).astype(int)
    converted_ends = np.round(all_bout_ends * conversion_ratio).astype(int)

    return converted_starts, converted_ends


def test_bout(all_bout_starts, all_bout_ends, theta):
    """
    Visualize and analyze the first 200 motion sequences (bouts) in the theta array.

    Args:
        all_bout_starts (array): Start indices of the bouts.
        all_bout_ends (array): End indices of the bouts.
        theta (array): Array containing angle data for each frame.
    """

    all_bout_starts = all_bout_starts[:200]  # Test first 200 bouts
    all_bout_ends = all_bout_ends[:200]

    for index, (start, end) in enumerate(zip(all_bout_starts, all_bout_ends)):
        plt.figure()
        start -= 10
        end += 10
        bout_data = theta[start:end]
        transposed_data = list(zip(*bout_data))

        for angle_data in transposed_data:

            x_original = np.arange(start, end)
            interp_func = interp1d(x_original, angle_data, kind='cubic')

            # Interpolated data
            x_new = np.linspace(start, end-1, num=(end-start)*10, endpoint=True)
            y_smoothed = interp_func(x_new)

            plt.plot(x_new, y_smoothed)

        plt.axvline(x=start+10, color='black', linestyle='--')
        plt.axvline(x=end-10, color='black', linestyle='--')
        plt.ylim(-7, 7)

        plt.xlabel('Frames')
        plt.ylabel('Interframe Angles')
        plt.title(f'Bout {index + 1}')
        # plt.savefig(f'bout_{index + 1}.png')
        plt.show()


if __name__ == '__main__':

    # data_path = '/Users/lanxinxu/Desktop/_makeBoutFiles/1_TailData/ES_10V_16x_Gcamp6s_mRubby_7dpf_42.h5'
    # data_path = '/Users/lanxinxu/Desktop/ES_10V_16x_Gcamp6s_mRubby_7dpf_42.h5'
    data_path = '/Users/lanxinxu/Desktop/UCL_Year_3/Epileptic_zebrafish_project/ZebraPoseMAE/gendata/preprocessing/test_data.h5'

    # Load data
    with h5py.File(data_path, 'r') as file:

        x = 'ds_x'
        y = 'ds_y'
        # theta = 'ds_theta'

        ds_x = np.array(file[x][...])
        ds_y = np.array(file[y][...])
        # ds_theta = np.array(file[theta][...])

    ds_x = ds_x[:, 1:] # discard the first point of every frame
    ds_y = ds_y[:, 1:]

    # Make a copy of original data as it's going to be changed
    ds_theta_copy = np.array(np.diff(ds_theta, axis=0))

    zebrafish = Zebrafish(ds_x, ds_y, ds_theta)

    # Interpolate data into 700Hz
    data_array, data_array_200hz, x_time, xi_time = zebrafish.interpolate_tail_data()
    interpolated_data = data_array[:, 0:8] * -1

    bc_size = 10  # The window size for boxcaf filter
    smoothed_data = zebrafish.tail_smoother(interpolated_data, bc_size)  # Filter data

    # Detect starts and ends of bouts in 700Hz
    all_bout_starts, all_bout_ends, threshold = zebrafish.bout_detector((smoothed_data))
    # Change 700Hz back into 200Hz
    all_bout_starts, all_bout_ends = convert_frames_to_200(all_bout_starts, all_bout_ends)

    np.save('bout_starts.npy', all_bout_starts)
    np.save('bout_ends.npy', all_bout_ends)

    # Plot first 200 bouts for test
    # Uncomment line 307 if you want to save figures
    test_bout(all_bout_starts, all_bout_ends, ds_theta_copy)

