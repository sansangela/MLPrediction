from src.social_lstm import SocialLSTM
from src.dataloader import DataLoader
from src.utils import mse_loss


import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop


def main():
    parser = argparse.ArgumentParser()
    # Device
    parser.add_argument(
        "--use_gpu", action="store_true", default=True, help="Use GPU if available"
    )

    # Dataset
    parser.add_argument(
        "--file_path",
        type=str,
        default="./data/eth/hotel/pixel_pos_interpolate.csv",
        help="File path for the dataset",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="eth_hotel", help="Name of the dataset"
    )
    parser.add_argument(
        "--file_path_processed",
        type=str,
        default="./data/preprocessed/",
        help="File path for processed data",
    )
    parser.add_argument(
        "--force_process_data_flag",
        action="store_true",
        default=False,
        help="Reprocess data from raw file if set",
    )

    # Model Architecture
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Dimensionality of LSTM hidden state",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Dimensionality of input embedding layer",
    )
    parser.add_argument(
        "--input_seq_length", type=int, default=8, help="Length of input sequence"
    )
    parser.add_argument(
        "--output_seq_length", type=int, default=12, help="Length of output sequence"
    )
    parser.add_argument(
        "--num_features", type=int, default=2, help="Number of input features"
    )

    # Hyperparameters
    parser.add_argument(
        "--learning_rate", type=float, default=0.003, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )  # TODO: change later
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")

    args = parser.parse_args()

    train(args)


def train(args):
    ## GPU CPU Selection
    device = "/CPU:0"
    if args.use_gpu and tf.config.list_physical_devices("GPU"):
        device = "/GPU:0"
        print("Using GPU...")

    # Load data
    dataloader = DataLoader(
        args.file_path,
        args.dataset_name,
        args.file_path_processed,
        args.force_process_data_flag,
    )
    dataloader.reset_frame_ptr()

    with tf.device(device):
        model = SocialLSTM()
        optimizer = RMSprop(args.learning_rate)

        for epoch in range(args.num_epochs):
            # For each batch
            epoch_loss = 0.0
            for batch in dataloader.generate_batches():
                inputs, targets = batch
                if not inputs or not targets:
                    # Traverse to the end
                    break

                batch_loss = 0.0

                # For each sequence
                for sequence_idx in range(args.batch_size):
                    input, target = inputs[sequence_idx], targets[sequence_idx]
                    (
                        dense_representation,
                        ped_id_to_index_map,
                    ) = dataloader.convert_to_dense_representation(input)
                    dense_representation_tf = tf.Variable(
                        tf.convert_to_tensor(dense_representation)
                    )
                    num_peds_per_frame = len(ped_id_to_index_map)

                    lstm_hidden = tf.Variable(
                        tf.zeros((num_peds_per_frame, args.hidden_dim))
                    )
                    lstm_cell = tf.Variable(
                        tf.zeros((num_peds_per_frame, args.hidden_dim))
                    )

                    with tf.GradientTape() as tape:
                        # Forward pass
                        out = model(dense_representation_tf, lstm_hidden, lstm_cell)
                        loss = mse_loss(target, out, ped_id_to_index_map) # TODO, placeholder
                        batch_loss += loss

                    # Compute gradients
                    grads = tape.gradient(loss, model.trainable_variables)

                    # Update parameters
                    trainable_variables = model.trainable_variables
                    optimizer.apply_gradients(zip(grads, trainable_variables))

                batch_loss /= args.batch_size
                epoch_loss += batch_loss

            print(
                "(epoch {}/{}), train_loss = {:.3f}".format(
                    epoch, args.num_epochs, epoch_loss
                )
            )


if __name__ == "__main__":
    main()
