import tensorflow as tf
from keras import layers as ly
import numpy as np

import CustomNN as customNN


# Class for the naive RNN model
class RNNCell(tf.keras.layers.Layer):

    def __init__(self, nodes_per_layer, layers, units, **kwargs):
        super(RNNCell, self).__init__(**kwargs)
        self.units = units                          # Units of the LSTM cell
        self.nodes_per_layer = nodes_per_layer      # Number of nodes per hidden layer
        self.layers = layers                        # Number of hidden layers used
        self.state_size = [self.units, self.units]  # Define state size based on the units used (for hidden and cell state)

        # Define and build LSTM cell and dense network,
        self.lstm_cell = None
        self.fNN = None
        self.build()

    def build(self):

        # Build FNN that predicts the stress based on the current hidden state
        self.fNN = customNN.getCustomFNN(hidden_layers=self.layers, nodes_per_layer=self.nodes_per_layer,
                                         activation_function='softplus', input_dim=self.units, output_dim=1,
                                         convex=False, bias=False, positivity=False)

        # Build LSTM cell with defined no of units using built-in LSTM function from keras
        self.lstm_cell = tf.keras.layers.LSTMCell(self.units)

        # Run initial input through LSTM cell to initialise dimensions
        inputs = tf.convert_to_tensor(np.zeros(shape=(1, 2)))
        h0 = tf.convert_to_tensor(np.zeros(shape=(1, self.units)))
        c0 = tf.convert_to_tensor(np.zeros(shape=(1, self.units)))
        _ = self.lstm_cell(inputs, [h0, c0])

    # Function to locally save the best weights during and after training
    def save_weights(self):

        self.fNN.save_weights('Weights_RNN/layers.weights.h5')

        weights = self.lstm_cell.get_weights()

        w1 = weights[0]
        w2 = weights[1]
        w3 = weights[2]

        np.save('Weights_RNN/lstm.w1.npy', w1)
        np.save('Weights_RNN/lstm.w2.npy', w2)
        np.save('Weights_RNN/lstm.w3.npy', w3)

    # Function to load previously saved weights
    def load_model(self):
        self.fNN.load_weights('Weights_RNN/layers.weights.h5')

        w1 = np.load('Weights_RNN/lstm.w1.npy')
        w2 = np.load('Weights_RNN/lstm.w2.npy')
        w3 = np.load('Weights_RNN/lstm.w3.npy')

        self.lstm_cell.set_weights([w1, w2, w3])

    # Call function of the RNN
    def call(self, inputs, states):

        # Get new hidden and cell state by calling the LSTM cell with the current input and previous states
        lstm_out, new_states = self.lstm_cell(inputs, states)

        # Input for FNN is the obtained hidden state
        x = lstm_out

        # Get stress prediction from the FNN based on hidden state
        y = self.fNN(x)

        return y, new_states


# Loss function: Mean squared error between true and predicted stress
def custom_loss(y_true, y_pred):
    loss = tf.keras.losses.mse((y_true[:, :, 0]), y_pred[:, :, 0])
    return loss


def main(x_train, y_train, train=True):
    # x_train contains [strain, dts]
    # y_train contains [stress]

    # Create model
    x = tf.keras.Input(shape=[None, 1])
    cell = RNNCell(nodes_per_layer=40, layers=3, units=16)
    network = ly.RNN(cell, return_sequences=True)
    y = network(x)
    model = tf.keras.Model(inputs=x, outputs=y)
    model.compile(optimizer='adam', loss=custom_loss)  # Adam optimiser, mean squared error loss

    if train:
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='Weights_RNN/best_model.weights.h5',  # Path where the weights will be saved
            monitor='loss',  # Metric to monitor
            save_best_only=True,  # Save only the best weights
            save_weights_only=True,  # Save only weights and not the entire model
            mode='min',  # Save the model with the minimum validation loss
            save_freq=20,  # Frequency at which the best weights are saved
            verbose=1  # Output messages
        )

        model.fit(x_train, y_train, epochs=16000, callbacks=[checkpoint_callback])  # Train model
        model.load_weights('Weights_RNN/best_model.weights.h5')                     # Load best weights during training
        cell.save_weights()                                                         # Save best weights

    # If model is not to be trained, load previous best weights
    else:
        cell.load_model()

    return model
