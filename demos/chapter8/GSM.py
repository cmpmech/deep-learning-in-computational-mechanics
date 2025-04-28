import tensorflow as tf
from keras import layers as ly
import numpy as np

import CustomNN as customNN


# Class for the GSM model
class GSM(tf.keras.layers.Layer):

    def __init__(self, nodes_per_layer, layers, units, s_eps, s_eps_dot, s_sig, **kwargs):
        super(GSM, self).__init__(**kwargs)
        self.units = units                          # Define number of units for the LSTM cell
        self.nodes_per_layer = nodes_per_layer      # Number of nodes per hidden layer
        self.layers = layers                        # Number of hidden layers used
        self.state_size = [units, units, 1]         # State size contains hidden and cell state, as well as previous internal variable gamma

        # Define different networks
        self.psi = None
        self.phi_plus = None
        self.phi_con = None

        self.fnn_h = None

        self.lstm_cell = None

        # Scaling factors for better training
        self.s_eps = s_eps              # Scaling factor for strain eps
        self.s_eps_dot = s_eps_dot      # Scaling factor for strain_rate eps_dot
        self.s_sig = s_sig              # Scaling factor for stress sig

        self.s_pot = self.s_eps * self.s_sig    # Scaling factor for potentials phi and psi

        self.build()

    def build(self):

        # FNN that outputs strain energy psi (or W)
        self.psi = customNN.getCustomFNN(hidden_layers=self.layers, nodes_per_layer=self.nodes_per_layer, activation_function='softplus',
                                         input_dim=2, output_dim=1, convex=False, bias=False)

        # positive NN that outputs positive part of dissipative potential phi+
        self.phi_plus = customNN.getCustomFNN(hidden_layers=self.layers, nodes_per_layer=self.nodes_per_layer, activation_function='softplus',
                                         input_dim=2, output_dim=1, convex=False, bias=True, positivity=True)

        # ICNN that outputs convex part of dissipative potential phi_con
        self.phi_con = customNN.getCustomFNN(hidden_layers=self.layers, nodes_per_layer=self.nodes_per_layer, activation_function='softplus',
                                         input_dim=1, output_dim=1, convex=True, bias=False)

        # FNN that predicts gamma based on hidden state of the LSTM cell
        self.fnn_h = customNN.getCustomFNN(hidden_layers=3, nodes_per_layer=16, activation_function='softplus',
                                         input_dim=self.units, output_dim=1, convex=False, bias=True, positivity=False)

        # LSTM cell
        self.lstm_cell = tf.keras.layers.LSTMCell(self.units)

        # Run initial input through LSTM cell to initialise dimensions
        inputs = tf.convert_to_tensor(np.zeros(shape=(1, 3)))
        h0 = tf.convert_to_tensor(np.zeros(shape=(1, self.units)))
        c0 = tf.convert_to_tensor(np.zeros(shape=(1, self.units)))
        _ = self.lstm_cell(inputs, [h0, c0])

    # Function to locally save the best weights during and after training
    def save_weights(self):
        self.psi.save_weights('Weights_GSM/psi.weights.h5')
        self.phi_con.save_weights('Weights_GSM/phi_con.weights.h5')
        self.phi_plus.save_weights('Weights_GSM/phi_plus.weights.h5')

        self.fnn_h.save_weights('Weights_GSM/h_fnn.weights.h5')

        weights = self.lstm_cell.get_weights()

        w1 = weights[0]
        w2 = weights[1]
        w3 = weights[2]

        np.save('Weights_GSM/lstm.w1.npy', w1)
        np.save('Weights_GSM/lstm.w2.npy', w2)
        np.save('Weights_GSM/lstm.w3.npy', w3)

        np.save('Weights_GSM/s_eps.npy', self.s_eps)
        np.save('Weights_GSM/s_eps_dot.npy', self.s_eps_dot)
        np.save('Weights_GSM/s_sig.npy', self.s_sig)

    # Function to load previously saved weights
    def load_model(self):
        self.psi.load_weights('Weights_GSM/psi.weights.h5')
        self.phi_con.load_weights('Weights_GSM/phi_con.weights.h5')
        self.phi_plus.load_weights('Weights_GSM/phi_plus.weights.h5')

        self.fnn_h.load_weights('Weights_GSM/h_fnn.weights.h5')

        w1 = np.load('Weights_GSM/lstm.w1.npy')
        w2 = np.load('Weights_GSM/lstm.w2.npy')
        w3 = np.load('Weights_GSM/lstm.w3.npy')

        self.lstm_cell.set_weights([w1, w2, w3])

        self.s_eps = np.load('Weights_GSM/s_eps.npy')
        self.s_eps_dot = np.load('Weights_GSM/s_eps_dot.npy')
        self.s_sig = np.load('Weights_GSM/s_sig.npy')
        self.s_pot = self.s_sig * self.s_eps

    # Call function of the GSM, only used during training not for predictions
    def call(self, inputs, states):

        # inputs are [strain, dts, stress]
        eps = tf.expand_dims(inputs[:, 0], -1)
        dt = tf.expand_dims(inputs[:, 1], -1)
        sig = tf.expand_dims(inputs[:, 2], -1)

        # Obtain previous output, pevious state and the previous internal variable gamma all from the last time step
        h_prev, c_prev, gamma_prev = states

        # Obtain new hidden and cell state from LSTM cell
        h, new_states = self.lstm_cell(inputs, [h_prev, c_prev])
        c = new_states[1]

        # Define zero inputs
        eps0 = tf.zeros_like(eps)
        sig0 = tf.zeros_like(sig)
        input0 = tf.concat([eps0, dt, sig0], axis=1)

        h0 = tf.zeros_like(h)
        c0 = tf.zeros_like(c)

        # Obtain hidden state for zero inputs
        h0, _ = self.lstm_cell(input0, [h0, c0])

        gamma = self.fnn_h(h) - self.fnn_h(h0)  # Obtain normalised internal variable gamma
        gamma_dot = (gamma - gamma_prev) / dt   # Obtain time derivative gamma_dot

        # Define zero inputs for internal variable
        gamma_dot0 = tf.zeros_like(gamma_dot)
        gamma0 = tf.zeros_like(gamma)

        # Scale input for better training
        eps = eps / self.s_eps
        sig = sig / self.s_sig
        gamma = gamma / self.s_eps
        gamma_dot = gamma_dot / self.s_eps_dot

        # Free Energy Potential W and stress
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(eps)
            tape.watch(gamma)
            W_NN = self.psi(tf.concat([eps, gamma], axis=1))    # Obtain strain energy W

            with tf.GradientTape(persistent=True) as tape0:
                tape0.watch(gamma0)
                tape0.watch(eps0)
                W_NN_0 = self.psi(tf.concat([eps0, gamma0], axis=1))    # Obtain strain energy W with zero inputs for normalisation

            # Normalise strain energy potential W
            W_i = W_NN - W_NN_0 - (tape0.gradient(W_NN_0, eps0) * eps) - (tape0.gradient(W_NN_0, gamma0) * gamma)
            W_i = W_i / self.s_pot  # Scale W

        # Internal stress tau is derivative of W w.r.t. gamma
        tau_i = (-1 * tape.gradient(W_i, gamma)) / self.s_sig

        # Stress is derivative of W w.r.t. eps
        sigma_i = (tape.gradient(W_i, eps)) / self.s_sig

        # Dissipation Potential
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(gamma_dot)
            phi_CON_NN = self.phi_con(gamma_dot)                            # Obtain convex part of phi
            phi_PLUS_NN = self.phi_plus(tf.concat([eps, gamma], axis=1))    # Obtain positive part of phi

            phi_NN = phi_PLUS_NN * phi_CON_NN   # Phi is multiplication of both parts

            with tf.GradientTape(persistent=True) as tape0:
                tape0.watch(gamma_dot0)
                phi_CON_NN_0 = self.phi_con(gamma_dot0)     # Obtain convex part with zero input for normalisation

                phi_NN_0 = phi_PLUS_NN * phi_CON_NN_0

            phi_i = phi_NN - phi_NN_0 - (tape0.gradient(phi_NN_0, gamma_dot0) * gamma_dot)  # Normalise phi
            phi_i = phi_i / self.s_pot                                                      # Scale phi

        # Internal stress tau_hat is derivative of phi w.r.t. gamma_dot
        tau_i_hat = (tape.gradient(phi_i, gamma_dot)) / self.s_sig

        # Put everything into one output
        combined_output = tf.concat([sigma_i, tau_i, tau_i_hat, gamma, gamma_dot], axis=1)

        return combined_output, [h, c, gamma*self.s_eps]

    # Function to predict new stresses using GSM model
    def predict(self, inputs, e=0.00001):

        # inputs are: [strain, dt]

        # Initialise internal variable as zero
        gamma = tf.zeros(shape=(1, 1))
        sig = []

        # Iterate over each time step in the sequence
        for n in range(len(inputs[0, :, 0])):

            # Obtain current strain and dt
            eps = tf.expand_dims(inputs[:, n, 0], -1)
            dt = tf.expand_dims(inputs[:, n, 1], -1)

            # Scale inputs
            eps = eps / self.s_eps
            gamma = gamma / self.s_eps

            # Initialise error margin for convergence
            R_i = e*1000

            # Initialise gamma_dot for each time step as zero
            gamma_dot = tf.zeros(shape=eps.shape)

            # Iterate as long as the error obtain from the Biot equation is larger than the specified margin
            # During each iteration of this while loop, gamma_dot is updated per Newton-Raphson
            while tf.abs(R_i) > e:

                # Initialise zero inputs
                gamma_dot0 = tf.zeros_like(gamma_dot)
                gamma0 = tf.zeros_like(gamma)
                eps0 = tf.zeros_like(eps)

                with tf.GradientTape(persistent=True) as tape_R:
                    tape_R.watch(gamma_dot)

                    # Free Energy Potential and stress
                    with tf.GradientTape(persistent=True) as tape:
                        tape.watch(eps)
                        tape.watch(gamma)
                        W_NN = self.psi(tf.concat([eps, gamma], axis=1))    # Obtain strain energy W

                        with tf.GradientTape(persistent=True) as tape0:
                            tape0.watch(gamma0)
                            tape0.watch(eps0)
                            W_NN_0 = self.psi(tf.concat([eps0, gamma0], axis=1))    # Obtain strain energy W with zero inputs for normalisation

                        # Normalise and scale W
                        W_i = W_NN - W_NN_0 - (tape0.gradient(W_NN_0, eps0) * eps) - (tape0.gradient(W_NN_0, gamma0) * gamma)
                        W_i = W_i / self.s_pot

                    # Obtain internal stress and stress as derivatives of W w.r.t. gamma and eps, respectively
                    tau_i = (-1 * tape.gradient(W_i, gamma)) / self.s_sig
                    sigma_i = (tape.gradient(W_i, eps)) / self.s_sig

                    # Dissipation Potential
                    with tf.GradientTape(persistent=True) as tape:
                        tape.watch(gamma_dot)
                        phi_CON_NN = self.phi_con(gamma_dot)                            # Convex part of phi
                        phi_PLUS_NN = self.phi_plus(tf.concat([eps, gamma], axis=1))    # Positive part of phi

                        phi_NN = phi_PLUS_NN * phi_CON_NN   # Phi is product of both parts

                        with tf.GradientTape(persistent=True) as tape0:
                            tape0.watch(gamma_dot0)
                            phi_CON_NN_0 = self.phi_con(gamma_dot0)     # Convex part with zero inputs for normalisation

                            phi_NN_0 = phi_PLUS_NN * phi_CON_NN_0

                        # Normalise and scale phi
                        phi_i = phi_NN - phi_NN_0 - (tape0.gradient(phi_NN_0, gamma_dot0) * gamma_dot)
                        phi_i = phi_i / self.s_pot

                    # Internal stress is derivative of phi w.r.t. gamma_dot
                    tau_i_hat = (tape.gradient(phi_i, gamma_dot)) / self.s_sig

                    # Calculate error from Biot equatoin
                    R_i = (tau_i - tau_i_hat)

                # If the error is smaller than the margin, add the stress prediction to the final output
                if tf.abs(R_i) < e:
                    print('Added: {}'.format(n))

                    # Update internal variable for new time step
                    gamma = gamma*self.s_eps + gamma_dot*self.s_eps_dot * dt

                    # Append stress to output
                    sig.append(sigma_i)

                # If error is still too large, update gamma_dot
                else:

                    # Update gamma_dot using Newton-Raphson and repeat while loop
                    dR_dzdot = tape_R.gradient(R_i, gamma_dot)
                    gamma_dot = gamma_dot - (tf.multiply(tf.linalg.inv(dR_dzdot), R_i))

                    # Error checking for divergence, if this occours during the prediction, then the training part of
                    # the Biot equation did not converge.
                    if tf.abs(gamma_dot) > 50000:
                        print("ERROR! Calculating Inverse failed --> Divergence!")
                        exit(-1)

        return tf.expand_dims(tf.concat(sig, 0), axis=0).numpy()


# Loss function: Mean squared error between true and predicted stress and Biot error
def custom_loss(y_true, y_pred):
    l_sig = tf.keras.losses.mse((y_true[:, :, 0]), y_pred[:, :, 0])

    l_biot = tf.keras.losses.mse(y_pred[:, :, 1], y_pred[:, :, 2])

    return 1.0 * l_sig + 25.0 * l_biot


def main(x_train, y_train, train=True):
    # x_train contains [strain, dts]
    # y_train contains [stress]

    # Obtain scaling factors for better training
    eps = x_train[:, :, 0]
    eps_dot = (x_train[:, 1:, 0] - x_train[:, :-1, 0]) / x_train[:, 1:, 1]
    sig = y_train[:, :, 0]
    s_eps = (0.5 * (tf.reduce_max(eps) - tf.reduce_min(eps)))
    s_eps_dot = (0.5 * (tf.reduce_max(eps_dot) - tf.reduce_min(eps_dot)))
    s_sig = (0.5 * (tf.reduce_max(sig) - tf.reduce_min(sig)))

    # Create model
    x = tf.keras.Input(shape=[None, 1])
    cell = GSM(nodes_per_layer=16, layers=3, units=16, s_eps=s_eps, s_eps_dot=s_eps_dot, s_sig=s_sig)
    network = ly.RNN(cell, return_sequences=True)
    y = network(x)
    model = tf.keras.Model(inputs=x, outputs=y)
    model.compile(optimizer='adam', loss=custom_loss)  # Adam optimiser, scaled sum of mean squared errors for loss

    # The training of the GSM model requires both strain, dts and stress as inputs
    x_train_mod = tf.concat([x_train, y_train], axis=2)

    if train:
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='Weights_GSM/best_model.weights.h5',  # Path where the weights will be saved
            monitor='loss',  # Metric to monitor
            save_best_only=True,  # Save only the best weights
            save_weights_only=True,  # Save only weights and not the entire model
            mode='min',  # Save the model with the minimum validation loss
            save_freq=20,  # Frequency at which the best weights are saved
            verbose=1  # Output messages
        )

        model.fit(x_train_mod, y_train, epochs=20000, callbacks=[checkpoint_callback])  # Train model
        model.load_weights('Weights_GSM/best_model.weights.h5')                         # Save best weights during training
        cell.save_weights()                                                             # Load best weights

    # If model is not to be trained, load previous best weights
    else:
        cell.load_model()

    return cell
