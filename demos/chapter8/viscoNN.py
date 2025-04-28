import tensorflow as tf
from keras import layers as ly

import CustomNN as customNN


# Class for the visco PANN model
class viscoPANN(tf.keras.layers.Layer):

    def __init__(self, nodes_per_layer, layers, **kwargs):
        super(viscoPANN, self).__init__(**kwargs)
        self.nodes_per_layer = nodes_per_layer      # Number of nodes per hidden layer
        self.layers = layers                        # Number of hidden layers used
        self.state_size = [1]                       # State size is only one 1D entry, as only the interal variable gamma is passed to the next time step

        # Define networks and trainable variables
        self.W_v = None
        self.W_e = None
        self.f_pNN = None

        self.alpha_v = None
        self.alpha_e = None

        self.beta_v = None
        self.beta_e = None

        self.build()

    def build(self):

        # ICNN that outputs strain energy W_v
        self.W_v = customNN.getCustomFNN(hidden_layers=self.layers, nodes_per_layer=self.nodes_per_layer, activation_function='softplus',
                                         input_dim=1, output_dim=1, convex=True, bias=True)

        # ICNN that outputs strain energy W_e
        self.W_e = customNN.getCustomFNN(hidden_layers=self.layers, nodes_per_layer=self.nodes_per_layer, activation_function='softplus',
                                         input_dim=1, output_dim=1, convex=True, bias=True)

        # positive NN that is needed to determine new internal variable gamma
        self.f_pNN = customNN.getCustomFNN(hidden_layers=self.layers, nodes_per_layer=self.nodes_per_layer, activation_function='softplus',
                                           input_dim=2, output_dim=1, convex=False, bias=True, positivity=True)

        # Following trainable variables alpha and beta are only used when ensuring the growth conditions, equation (1.6)

        # Variable alpha for strain energy W_v
        self.alpha_v = tf.Variable(initial_value=1.0, constraint=tf.keras.constraints.NonNeg(), dtype=tf.float32)

        # Variable alpha for strain energy W_e
        self.alpha_e = tf.Variable(initial_value=1.0, constraint=tf.keras.constraints.NonNeg(), dtype=tf.float32)

        # Variable beta for strain energy W_v
        self.beta_v = tf.Variable(initial_value=1.0, constraint=tf.keras.constraints.NonNeg(), dtype=tf.float32)

        # Variable beta for strain energy W_e
        self.beta_e = tf.Variable(initial_value=1.0, constraint=tf.keras.constraints.NonNeg(), dtype=tf.float32)

    # Function to locally save the best weights during and after training
    def save_weights(self):

        self.W_v.save_weights('Weights_PANN/W_v.weights.h5')
        self.W_e.save_weights('Weights_PANN/W_e.weights.h5')
        self.f_pNN.save_weights('Weights_PANN/f_pNN.weights.h5')

    # Function to load previously saved weights
    def load_model(self):
        self.W_v.load_weights('Weights_PANN/W_v.weights.h5')
        self.W_e.load_weights('Weights_PANN/W_e.weights.h5')
        self.f_pNN.load_weights('Weights_PANN/f_pNN.weights.h5')

    # Call function of the visco-PANN
    def call(self, inputs, states):

        # inputs are [strain, dts]
        eps = tf.expand_dims(inputs[:, 0], -1)
        dt = tf.expand_dims(inputs[:, 1], -1)

        eps0 = tf.zeros_like(eps)

        gamma = states[0]   # Previous internal variable

        with tf.GradientTape(persistent=True) as tape0:
            tape0.watch(eps0)
            W_v0 = self.W_v(eps0 - gamma)   # Strain energy W_v for zero strain input
            W_e0 = self.W_e(eps0)           # Strain energy W_e for zero strain input

        dW_v0 = tape0.gradient(W_v0, eps0)  # Derivate of W_v0 w.r.t. eps
        dW_e0 = tape0.gradient(W_e0, eps0)  # Derivate of W_e0 w.r.t. eps

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(eps)

            # Get strain energies W_v and W_e. Not that only the normalisation using W_v0 and W_e0 (ensuring zero strain
            # energy for a zero-strain input are used. The other possible normalisation are commented out and not used
            W_v = self.W_v(eps - gamma) - W_v0   # - (dW_v0 * (eps - gamma)) + self.alpha_v * ((1.0 / (1.0 + eps)) + eps - 1.0) + self.beta_v * (eps**2)
            W_e = self.W_e(eps) - W_e0   # - (dW_e0 * eps) + self.alpha_e * ((1.0 / (1.0 + eps)) + eps - 1.0) + self.beta_e * (eps**2)

            W = W_v + W_e   # Total strain energy

        dW_v = tape.gradient(W_v, eps)  # Derivative of strain energy W_v w.r.t. eps
        dW_e = tape.gradient(W_e, eps)  # Derivative of strain energy W_e w.r.t. eps

        sigma = tape.gradient(W, eps)  # Obtain stress as derivative of strain energy w.r.t. eps

        # Obtain new internal variable
        gamma_new = gamma + dt * self.f_pNN(tf.concat([eps, gamma], axis=1)) * dW_v

        return sigma, [gamma_new]


# Loss function: Mean squared error between true and predicted stress
def custom_loss(y_true, y_pred):
    loss = tf.keras.losses.mse((y_true[:, :, 0]), y_pred[:, :, 0])
    return loss


def main(x_train, y_train, train=True):
    # x_train contains [strain, dts]
    # y_train contains [stress]

    # Create model
    x = tf.keras.Input(shape=[None, 1])
    cell = viscoPANN(nodes_per_layer=16, layers=1)
    network = ly.RNN(cell, return_sequences=True)
    y = network(x)
    model = tf.keras.Model(inputs=x, outputs=y)
    model.compile(optimizer='adam', loss=custom_loss)   # Adam optimiser, mean squared error loss

    if train:
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='Weights_PANN/best_model.weights.h5',  # Path where the weights will be saved
            monitor='loss',  # Metric to monitor
            save_best_only=True,  # Save only the best weights
            save_weights_only=True,  # Save only weights and not the entire model
            mode='min',  # Save the model with the minimum validation loss
            save_freq=20,  # Frequency at which the best weights are saved
            verbose=1  # Output messages
        )

        model.fit(x_train, y_train, epochs=12000, callbacks=[checkpoint_callback])  # Train model
        model.load_weights('Weights_PANN/best_model.weights.h5')                    # Load best weights during training
        cell.save_weights()                                                         # Save best weights

    # If model is not to be trained, load previous best weights
    else:
        cell.load_model()

    return model
