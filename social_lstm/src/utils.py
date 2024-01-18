import tensorflow as tf
import numpy as np

def mse_loss(y_true, y_pred, ped_id_to_index_map):
    """
    Calculate the Mean Squared Error (MSE) loss between predicted and true positions.

    Args:
        y_true (list): True pedestrian positions of shape (num_sequences, np.array(num_pedestrians, 3)).
        y_pred (tf.Tensor): Predicted pedestrian positions of shape (num_sequences, num_pedestrians, 2).
        ped_id_to_index_map (dict): Dictionary mapping pedestrian IDs to their indices.

    Returns:
        tf.Tensor: Mean Squared Error loss.
    """
    num_seq, num_ped = len(y_true), len(ped_id_to_index_map)
    y_true_dense_representation = np.zeros((num_seq, num_ped, 2))
    for sequence_idx in range(len(y_true)):
        indices = [ped_id_to_index_map[x] for x in y_true[sequence_idx][:, 0] if x in ped_id_to_index_map.keys()]
        if not indices:
            continue
        y_true_dense_representation[sequence_idx, indices, :] = y_true[sequence_idx][:, 1:3]
    
    return -tf.math.reduce_mean(y_true_dense_representation-y_pred)