import tensorflow as tf


# Function to create custom FNNs for the model classes
def getCustomFNN(hidden_layers=1, nodes_per_layer=4, activation_function='softplus', input_dim=1, output_dim=1,
                 convex=False, bias=True, positivity=False):

    # Setup model
    model = tf.keras.Sequential()

    # Define model input shape
    model.add(tf.keras.Input(shape=(input_dim,)))

    # Add hidden layers
    for i in range(hidden_layers):
        if convex:
            if i == 0:
                model.add(tf.keras.layers.Dense(nodes_per_layer, activation=activation_function, use_bias=bias))
            else:
                model.add(tf.keras.layers.Dense(nodes_per_layer, activation=activation_function, use_bias=bias,
                                            kernel_constraint=tf.keras.constraints.NonNeg()))
        else:
            model.add(tf.keras.layers.Dense(nodes_per_layer, activation=activation_function, use_bias=bias))

    # Add output layer
    if convex:
        model.add(tf.keras.layers.Dense(output_dim, activation='linear', use_bias=bias,
                                            kernel_constraint=tf.keras.constraints.NonNeg()))
    elif positivity:
        model.add(tf.keras.layers.Dense(output_dim, activation='softplus', use_bias=bias))

    else:
        model.add(tf.keras.layers.Dense(output_dim, activation='linear', use_bias=bias))

    return model
