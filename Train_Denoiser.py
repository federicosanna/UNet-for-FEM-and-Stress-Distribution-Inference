# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

import os
import argparse

import matplotlib.pyplot as plt

import keras


def train_denoiser(denoise_generator_poly, denoise_generator_val_poly, model, epochs):

    epochs = epochs
    denoise_model = model

    # Create the folders where to save the trained models and the training plots
    if not os.path.exists('./Saved_Models'):
        os.mkdir('./Saved_Models')
    if not os.path.exists('./Saved_Training_Plots'):
        os.mkdir('./Saved_Training_Plots')
    # Construct the argument parser and parse the arguments
    # Allow naming your saved model in different ways without changing the code
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", type=str, required=False,
                    help="(Still to be implemented)path dataset of input images")
    ap.add_argument("-m", "--model", type=str, default='./Saved_Models/modellino.model',
                    help="path to folder where to save trained model (ex.: --model Saved_Models/model01.model)")
    ap.add_argument("-p", "--plot", type=str, default="./Saved_Training_Plots/plot.png",
                    help="path to output loss/accuracy plot (ex.: --model Saved_Training_Plots/plot01.png)")
    args = vars(ap.parse_args())

    sgd = keras.optimizers.SGD(lr=0.00001, momentum=0.9, nesterov=True)
    adag = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    denoise_model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['mae'])

    ### Use a loop to save for each epoch the weights in an external website in
    ### case colab stops. Every time you call fit/fit_generator the weigths are NOT
    ### reset, so e.g. calling 5 times fit(epochs=1) behave as fit(epochs=5)
    acc_denoise_history = []
    val_acc_denoise_history = []
    for e in range(epochs):
        print("Epoch is " + str(e))
        denoise_history = denoise_model.fit_generator(generator=denoise_generator_poly,
                                                      epochs=1, verbose=1,
                                                      validation_data=denoise_generator_val_poly)
        ### Saves optimizer and weights
        denoise_model.save('denoise.h5')
        ### Uploads files to external hosting
        # curl - F "file=@denoise.h5" https: // file.io
        # list all data in history
        print(denoise_history.history.keys())
        acc_denoise_history.append(denoise_history.history['loss'][0])
        val_acc_denoise_history.append(denoise_history.history['val_loss'][0])
        print("Length of acc_descriptor_history is:" + str(len(acc_denoise_history)))
        print(acc_denoise_history)
        print("Length of val_acc_descriptor_history is:" + str(len(val_acc_denoise_history)))
        print(val_acc_denoise_history)

    # save the network to disk
    print("[INFO] serializing network to '{}'...".format(args["model"]))
    model.save(args["model"])

    # summarize history for accuracy
    plt.plot(acc_denoise_history)
    plt.plot(val_acc_denoise_history)
    plt.title('denoiser loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args["plot"])
    plt.show()
    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # Print the vectors of interest
    print(acc_denoise_history)
    print(len(acc_denoise_history))
    print(val_acc_denoise_history)
    print(len(val_acc_denoise_history))