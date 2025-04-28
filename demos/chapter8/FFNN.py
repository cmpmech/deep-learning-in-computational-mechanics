import tensorflow as tf
from keras import layers as ly

import CustomNN as customNN


# Class for the naive FFNN model
class FFNN(tf.keras.layers.Layer):

    def __init__(self, nodes_per_layer, layers, no_of_prev_steps, fix_psi=False, fix_eta=False, **kwargs):
        super(FFNN, self).__init__(**kwargs)
        self.feature_dim = 3                        # Feature dimension of the state = 3 as for each time step, 3 values saved in states: [strain, stress, dts]

        self.nodes_per_layer = nodes_per_layer      # Number of nodes per hidden layer
        self.layers = layers                        # Number of hidden layers used
        self.no_of_prev_steps = no_of_prev_steps    # Number of previous time steps taken into account to determine new stress

        self.state_size = []                        # Size of state depends on the number of previous time steps taken into account
        for _ in range(self.feature_dim):
            for _ in range(no_of_prev_steps):
                self.state_size.append(1)

        # Define and build NN
        self.fNN = None
        self.build()

    def build(self):

        # Build NN
        self.fNN = customNN.getCustomFNN(hidden_layers=self.layers, nodes_per_layer=self.nodes_per_layer,
                                         activation_function='softplus',
                                         input_dim=2+(self.no_of_prev_steps*self.feature_dim), output_dim=1,
                                         convex=False, bias=False, positivity=False)

    # Function to locally save the best weights during and after training
    def save_weights(self):

        self.fNN.save_weights('Weights_FNN/layers.weights.h5')

    # Function to load previously saved weights
    def load_model(self):
        self.fNN.load_weights('Weights_FNN/layers.weights.h5')

    # Call function of the NN
    def call(self, inputs, states):

        x = inputs # Input contains [strain, dts]

        # Input for NN is the current x and the stress, strain and dt of previous time-steps found in states[i]
        for i in range(self.feature_dim*self.no_of_prev_steps):
            x = tf.concat([x, states[i]], axis=1)

        # Get new stress
        y = self.fNN(x)

        # Save current stress, strain and dt in states
        new_states = [y, tf.expand_dims(inputs[:, 0], -1), tf.expand_dims(inputs[:, 1], -1)]

        # Append (previous time-steps - 1) in states
        for i in range(self.feature_dim*(self.no_of_prev_steps-1)):
            new_states.append(states[i])

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
    cell = FFNN(nodes_per_layer=40, layers=5, no_of_prev_steps=3)
    network = ly.RNN(cell, return_sequences=True)
    y = network(x)
    model = tf.keras.Model(inputs=x, outputs=y)
    model.compile(optimizer='adam', loss=custom_loss)  # Adam optimiser, mean squared error loss

    if train:
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='Weights_FNN/best_model.weights.h5',  # Path where the weights will be saved
            monitor='loss',  # Metric to monitor
            save_best_only=True,  # Save only the best weights
            save_weights_only=True,  # Save only weights and not the entire model
            mode='min',  # Save the model with the minimum validation loss
            save_freq=20,  # Frequency at which the best weights are saved
            verbose=1  # Output messages
        )

        model.fit(x_train, y_train, epochs=16000, callbacks=[checkpoint_callback])  # Train model
        model.load_weights('Weights_FNN/best_model.weights.h5')                     # Load best weights during training
        cell.save_weights()                                                         # Save best weights

    # If model is not to be trained, load previous best weights
    else:
        cell.load_model()

    return model
