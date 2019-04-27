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
    n_gons = [4, 5, 6, 7, 8]  # Types of polygons to be contained in the dataset
    canvas_size = 64
    size_of_ds_poly = 6000

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

    ds_poly = GP.SlighlyMoreClevr(n_gons=n_gons, canvas_size=canvas_size, size_of_ds_poly=size_of_ds_poly)  # Generate dataset
    random_indices_poly = torch.randperm(len(ds_poly))
    generator = GP.DenoiseHPatchesPoly(random_indices_poly=random_indices_poly, ds_poly=ds_poly, batch_size=50)
    imgs, imgs_clean = next(iter(generator))
    index = np.random.randint(0, imgs.shape[0])
    imgs_den = denoise_model.predict(imgs)
    plt.subplot(131)
    plt.imshow(imgs[index,:,:,0], cmap='gray')
    plt.title('Noisy', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(132)
    plt.imshow(imgs_den[index,:,:,0], cmap='gray')
    plt.title('Denoised', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(133)
    plt.imshow(imgs_clean[index,:,:,0], cmap='gray')
    plt.title('Clean', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.show()


if __name__ == '__main__':
    plot_sample()