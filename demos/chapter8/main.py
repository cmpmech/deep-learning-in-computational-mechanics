import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import data as ld
import FFNN as fnn
import RNN as rnn
import viscoNN as visc
import GSM as gsm


# Function to test the chosen model. Specify the model_type in the main function
def test_model(model_type='gsm', train=True):

    # Generate training data
    i = 0
    for strain_rate in [5e-2, 5e-3]:    # Generate load paths for different strain rates
        pt, pe, ps = ld.evp1d_dlcm2(strain_rate)  # pt: time, pe:strain, ps:stress

        # Calculate dt based on times pt
        pdt = pt[1:, :] - pt[:-1, :]
        pdt = tf.concat([tf.expand_dims(pdt[0, :], axis=0), pdt], axis=0)

        # Adjust dimensions
        pdt = tf.expand_dims(pdt, axis=0)
        pe = tf.expand_dims(pe, axis=0)
        ps = tf.expand_dims(ps, axis=0)

        # Append load path to training data
        x_train_part = tf.concat([pe, pdt], axis=2)
        y_train_part = ps

        if i == 0:
            x_train = x_train_part
            y_train = y_train_part
        else:
            x_train = tf.concat([x_train, x_train_part], axis=0)
            y_train = tf.concat([y_train, y_train_part], axis=0)

        i = i + 1

    # Generate model based on chosen model
    match model_type:
        case 'FNN':
            model = fnn.main(x_train, y_train, train=train)

        case 'RNN':
            model = rnn.main(x_train, y_train, train=train)

        case 'visco-PANN':
            model = visc.main(x_train, y_train, train=train)

        case 'GSM':
            model = gsm.main(x_train, y_train, train=train)

        case _:
            print('Chosen model does not exist')
            exit(1)

    # Evaluate test cases
    i = 0
    for strain_rate in [5e-2, 5e-3, 1e-1, 1e-2, 1e-3]:
        pt, pe, ps = ld.evp1d_dlcm2(strain_rate)  # pt: time, pe:strain, ps:stress

        pdt = pt[1:, :] - pt[:-1, :]
        pdt = tf.concat([tf.expand_dims(pdt[0, :], axis=0), pdt], axis=0)

        pdt = tf.expand_dims(pdt, axis=0)
        pe = tf.expand_dims(pe, axis=0)
        ps = tf.expand_dims(ps, axis=0)

        x_train = tf.concat([pe, pdt], axis=2)
        y_train = ps

        pred = model.predict(x_train)

        plt.figure(i, dpi=300)
        plt.plot(x_train[0, :, 0], y_train[0, :, 0])
        plt.plot(x_train[0, :, 0], pred[0, :, 0])
        plt.xlabel('Strain')
        plt.ylabel('Stress')
        plt.title('Strain rate: {}'.format(strain_rate))
        plt.legend(('True stress', 'Predicted stress'))

        # Save plots and data. Adjust save path accordingly to chosen model
        plt.savefig('data/{}/plot_strain_0.3_rate_{}.png'.format(model_type, strain_rate))
        with open('data/{}/strain_0.3_rate_{}.txt'.format(model_type, strain_rate), 'w') as output:

            for j in range(len(pred[0, :, 0])):
                output.write(np.array2string(pt[j, 0].numpy()) + ' ' + np.array2string(x_train[0, j, 0].numpy()) + ' '
                             + np.array2string(pred[0, j, 0]) + ' ' + np.array2string(y_train[0, j, 0].numpy()))
                output.write("\n")
        i = i + 1

    plt.show()


if __name__ == '__main__':
    ##################################################
    # CHOOSE MODEL
    # FNN: Naive feed forward NN
    # RNN: Naive RNN with LSTM cell
    # visco-PANN: Visco-PANN model
    # GSM: GSM model
    ##################################################

    # model = 'FNN'
    # model = 'RNN'
    model = 'visco-PANN'
    # model = 'GSM'

    # Chose whether to train the model from scratch or to use the last existing model and only predict the test cases
    train = True

    test_model(model, train=train)
