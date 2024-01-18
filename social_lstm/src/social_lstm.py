import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, LSTMCell
from tensorflow.keras.models import Model


class SocialLSTM(Model):
    def __init__(
        self,
        embedding_dim=64,
        hidden_dim=128,
        pool_size=(8, 8),
        neighbors=32,
        num_features=2,
        device="/GPU:0",
    ):
        """
        Initialize the SocialLSTM model.

        Args:
            embedding_dim (int, optional): Dimensionality of the input and social embeddings. Defaults to 64.
            hidden_dim (int, optional): Dimensionality of the LSTM hidden state. Defaults to 128.
            pool_size (tuple, optional): Size of the spatial pooling window. Defaults to (8,8).
            neighbors (int, optional): Number of neighbors to consider. Defaults to 32.
            num_features (int, optional): Number of output features (y_coordinates, x_coordinates). Defaults to 2.
        """
        super(SocialLSTM, self).__init__()

        # Device
        if device == "/GPU:0" and tf.config.list_physical_devices("GPU"):
            self.device = device
        else:
            self.device = "/CPU:0"

        # Layers
        self.input_embedding_layer = Dense(embedding_dim)
        self.social_embedding_layer = Dense(embedding_dim)  # TODO
        self.lstm = LSTMCell(hidden_dim)
        self.output_embedding_layer = Dense(num_features)
        # self.embedding_layer = Embedding(input_dim=num_features, output_dim=embedding_dim)
        # self.spatial_pooling = AveragePooling2D(pool_size=pool_size, strides=strides)

        # Activation Function
        self.relu = ReLU()

    def call(self, input, lstm_hidden, lstm_cell):
        """
        Forward pass of the SocialLSTM model.

        Args:
            input (tf.Tensor): Input tensor of shape (sequence_length, num_ped, num_features).
            lstm_hidden (tf.Tensor): LSTM hidden state.
            lstm_cell (tf.Tensor): LSTM cell state.

        Returns:
            out (tf.Tensor): Output tensor of shape (sequence_length, num_ped, num_features).
        """
        # Initialize an empty list to store outputs for each time step
        out_split = tf.split(
            tf.zeros(input.shape), num_or_size_splits=input.shape[0], axis=0
        )

        # Iterate through each time step in the input sequence
        for frame_id in range(input.shape[0]):
            # Get the input and apply input embedding
            input_t = input[frame_id]
            input_embed_out = self.input_embedding_layer(input_t)
            input_embed_out = self.relu(input_embed_out)

            # TODO: Apply social embedding
            social_embed_out = self.social_embedding_layer(input_t)  # Placeholder
            social_embed_out = self.relu(social_embed_out)

            # Concatenate input and social embeddings
            embed_out = tf.concat([input_embed_out, social_embed_out], axis=1)

            # Apply LSTM layer
            lstm_out, new_state = self.lstm(embed_out, (lstm_hidden, lstm_cell))
            lstm_hidden, lstm_cell = new_state

            # Apply output embedding
            out_t = self.output_embedding_layer(lstm_out)
            out_t = tf.expand_dims(out_t, axis=0)
            out_split[frame_id] = out_t

        out = tf.concat(out_split, axis=0)
        return out
