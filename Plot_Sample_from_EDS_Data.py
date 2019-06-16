import numpy as np

# It is important to import Generate_Poly before changing the backend of matplotlib otherwise it would break
import Generate_Poly as GP
import matplotlib

gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
print("Using:",matplotlib.get_backend())

import matplotlib.pyplot as plt
import matplotlib
import torch

from keras.models import load_model


import argparse


def plot_sample():
    """Can be run from command window to plot a noisy patch, denoised patch and clean patch.
    Args:
        None, but can be modified to take:
            denoise_model: keras model to predict clean patch
    To run from command window:
    $ python Plot_Sample.py
    OR to run with a particular model
    $ python Plot_Sample.py  --model Saved_Models/modellino.model
    OR
    $ python -c "import Plot_Sample; print(Plot_Sample.plot_sample())"

    """
    canvas_size = 64

    # Construct the argument parser and parse the arguments
    # Allow naming your saved model in different ways without changing the code
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=False,
                    help="path to out input directory of images")
    ap.add_argument("-m", "--model", default='./Saved_Models/modellino.model',
                    help="path to pre-trained model")
    args = vars(ap.parse_args())

    # load the pre-trained network
    print("[INFO] loading pre-trained network...")
    denoise_model = load_model(args["model"])

    # ====================== Input & Output are from same dataset =========================
    # Inputs = np.load(
    #     '/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Labels_DataSet.npy')
    # random_indices_poly = torch.randperm(len(Inputs))
    # generator = GP.DenoiseHPatchesPoly_Stage_0(random_indices_poly=random_indices_poly, ds_poly=Inputs, batch_size=50)

    # ====================== Input & Output are from different dataset =========================
    Inputs = np.load(
        '/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Params_DataSet.npy')
    Labels = np.load(
        '/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Labels_DataSet.npy')
    random_indices_poly = torch.randperm(len(Inputs))
    generator = GP.DenoiseHPatchesPoly_Exp5(random_indices_poly=random_indices_poly, inputs=Inputs, labels=Labels, batch_size=50)


    imgs, imgs_clean = next(iter(generator))
    index = np.random.randint(0, imgs.shape[0])
    imgs_den = denoise_model.predict(imgs)
    plt.subplot(131)
    plt.imshow(imgs[index,:,:,0], cmap='gray')
    plt.title('Input', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(132)
    plt.imshow(imgs_den[index,:,:,0], cmap='gray')
    plt.title('Reconstructed', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(133)
    plt.imshow(imgs_clean[index,:,:,0], cmap='gray')
    plt.title('GT', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.show()


if __name__ == '__main__':
    plot_sample()