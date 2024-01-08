import pandas
import numpy as np
import os


def save_skeleton_files(data, directory):
    """
    Saves each bout as a .skeleton file with a custom naming structure.

    :param data: Numpy array of shape (34015, 82, 19, 3) containing all bouts.
    :param directory: Directory where the .skeleton files will be saved.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    num_fish = 1  # Assuming all bouts are from the same fish
    num_bouts = data.shape[0]
    num_frames = data.shape[2]
    num_joints = data.shape[3]

    skeleton_file_list = []  # List to store the file names

    for bout_index in range(num_bouts):
        file_name = f"fish{num_fish:05d}bout{bout_index + 1:05d}"
        skeleton_file_path = os.path.join(directory, file_name + ".skeleton")

        with open(skeleton_file_path, 'w') as file:
            # Write the number of frames
            file.write(f"{num_frames}\n")

            # Write the joint data for each frame
            for frame in range(num_frames):
                # Add num_fish and num_joints before each frame's data
                file.write(f"{num_fish}\n{num_joints}\n")

                for joint in range(num_joints):
                    x, y = data[bout_index, frame, joint, :2]
                    file.write(f"{x} {y}\n")

        # Add the file name (without extension) to the list
        skeleton_file_list.append(file_name)
        print(f'bout: {bout_index}')

    # Write the file names to a .txt file
    list_file_path = "../poseRdata_82/statistics/skes_available_name.txt"
    with open(list_file_path, 'w') as list_file:
        for file_name in skeleton_file_list:
            list_file.write(file_name + "\n")


if __name__ == '__main__':

    metadata = pandas.read_hdf(
        '/Users/lanxinxu/Desktop/INTERN_2023/PycharmProjects/PoseR/poser_dataset/ZebTensor/bout_metadata.h5')
    quality = metadata['quality']
    good_indices = quality[quality == 'good'].index
    good_indices_list = good_indices.tolist()

    fishdata = np.load(
        '/Users/lanxinxu/Desktop/INTERN_2023/PycharmProjects/PoseR/poser_dataset/ZebTensor/bouts.npy')
    fishdata = np.squeeze(fishdata, axis=-1)  # 移除尺寸为1的维度，因为只有一条鱼，不需要这个维度
    fishdata = fishdata[good_indices_list, :, :]  # 只取good quality的回合
    fishdata = fishdata.transpose(0, 2, 3, 1)

    save_directory = '../poseRdata_82/raw_data/skeleton_files'
    save_skeleton_files(fishdata, save_directory)

    print('Finished.')


