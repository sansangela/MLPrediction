import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle


class DataLoader:
    def __init__(
        self,
        file_path,
        dataset_name,
        data_preprocessed_path,
        force_process_data_flag=False,
        batch_size=5,
        input_seq_length=8,
        output_seq_length=12,
        is_train=True,
        train_dev_split=[1.0, 0.0],
    ):
        """Constructor for the DataLoader class.

        Args:
            file_path (str): Path to the data file.
            dataset_name (str): Name of the dataset.
            data_preprocessed_path (str): Path to store preprocessed data.
            force_process_data_flag (bool, optional): Flag to force data preprocessing from raw.
            batch_size (int, optional): Size of batches for training. Defaults to 5.
            input_seq_length (int, optional): Length of input sequence. Defaults to 8.
            output_seq_length (int, optional): Length of output sequence. Defaults to 12.
            is_train (bool, optional): Flag indicating if it's for training. Defaults to True.
            train_dev_split (list, optional): Train-validation split ratio. Defaults to [1.0, 0.0].
        """

        self.file_path = file_path
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.seq_length = input_seq_length + output_seq_length
        self.data_preprocessed_path = data_preprocessed_path
        self.is_train = is_train
        self.train_fraction = train_dev_split[0]
        self.val_fraction = train_dev_split[1]

        self.frame_ptr = 0

        self.preprocess_data(force_process_data_flag)

    def generate_batches(self):
        """
        Generator function to yield data batches.

        Yields:
            tuple: A tuple containing input data and target data.
                The shape of both input and target data is (batch_size, seq_length, np.array(num_peds_per_frame, feature_dim=3)).
        """
        # TODO: aggregate dataset
        num_seq = len(self.data) // self.seq_length
        num_batches = num_seq // self.batch_size

        for _ in range(num_batches):
            batch_idx = 0
            data_x = []
            data_y = []
            while batch_idx < self.batch_size:
                frame_start = self.frame_list[self.frame_ptr]
                if frame_start + self.seq_length < len(self.frame_list):
                    batch_data_x = self.data[
                        frame_start : frame_start + self.seq_length
                    ]
                    batch_data_y = self.data[
                        frame_start + 1 : frame_start + 1 + self.seq_length
                    ]
                    batch_ped_indices = self.ped_indices[
                        frame_start : frame_start + self.seq_length
                    ]

                    data_x.append(batch_data_x)
                    data_y.append(batch_data_y)

                    self.frame_ptr += self.seq_length
                    batch_idx += 1
                else:
                    break

            # data: (batch_size, seq_length, np.array(num_peds_per_frame,feature_dim=3))
            yield (data_x, data_y)

    def preprocess_data(self, force_process_data_flag=False):
        """Preprocesses the data by loading or generating it.

        Args:
            force_process_data_flag (bool, optional): Flag to force data preprocessing from raw.
        """
        self.data_preprocessed_file = (
            self.data_preprocessed_path + self.dataset_name + "_data.pkl"
        )

        # Read and Preprocess Data
        if os.path.exists(self.data_preprocessed_file) and not force_process_data_flag:
            print("Load preprocessed data")
            with open(self.data_preprocessed_file, "rb") as file:
                self.data, self.ped_indices, self.frame_list = pickle.load(file)
        else:
            print("Preprocess data")
            self.preprocess_data_from_raw(force_process_data_flag)

        print("Preprocess data")
        self.preprocess_data_from_raw(force_process_data_flag)
        print("Done preprocessing")

    def preprocess_data_from_raw(self, force_process_data_flag=False):
        """Preprocesses raw data and stores it.

        Args:
            force_process_data_flag (bool, optional): Flag to force data preprocessing from raw.
        """
        orig_df = pd.read_csv(self.file_path, header=None)
        frame_numbers = orig_df.iloc[0].astype(int)
        pedestrian_ids = orig_df.iloc[1].astype(int)
        y_coordinates = orig_df.iloc[2].astype(float)
        x_coordinates = orig_df.iloc[3].astype(float)
        df = pd.DataFrame(
            {
                "frame_id": frame_numbers,
                "pedestrian_id": pedestrian_ids,
                "y": y_coordinates,
                "x": x_coordinates,
            }
        )
        df = df.drop_duplicates(subset=["frame_id", "pedestrian_id"])

        grouped = df.groupby("frame_id")
        ## TODO: process all datasets
        self.ped_indices = []
        self.frame_list = []
        self.data = []

        for frame_id, group in grouped:
            ped_list = group["pedestrian_id"].tolist()
            self.frame_list.append(frame_id)
            self.ped_indices.append(ped_list)
            self.data.append(np.array(group[["pedestrian_id", "y", "x"]]))

        ## TODO: store
        if not os.path.exists(self.data_preprocessed_file) or force_process_data_flag:
            with open(self.data_preprocessed_file, "wb") as file:
                pickle.dump((self.data, self.ped_indices, self.frame_list), file)
            print(f"Write preprocessed data to {self.data_preprocessed_file}")

    def reset_frame_ptr(self, frame_ptr=0):
        """Resets the frame pointer.

        Args:
            frame_ptr (int, optional): Frame pointer value. Defaults to 0.
        """
        self.frame_ptr = frame_ptr


if __name__ == "__main__":
    print("*****Dataloader Preprocess Data*****")
    file_path = "/root/Projects/MLPrediction/data/eth/hotel/pixel_pos_interpolate.csv"
    dataset_name = "eth_hotel"
    file_path_processed = "/root/Projects/MLPrediction/data/preprocessed/"
    force_process_data_flag = True

    dataloader = DataLoader(
        file_path, dataset_name, file_path_processed, force_process_data_flag
    )

    for batch in dataloader.generate_batches():
        inputs, targets = batch
