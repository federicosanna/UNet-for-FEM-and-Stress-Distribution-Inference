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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    # n_gons = [4, 5, 6, 7, 8]  # Types of polygons to be contained in the dataset
    n_gons = [5]
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
    Shallow_model = load_model('./Saved_Models/ShallowNorm.model')
    Baseline_model = load_model('./Saved_Models/BaselineForce.model')
    Full_model = load_model('./Saved_Models/FullNorm.model')
    Half_model = load_model('./Saved_Models/HalfForce.model')

    Inputs = np.load(
        '/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Params_DataSet.npy')
    Labels = np.load(
        '/media/federico/Seagate Expansion Drive/DataProject/EDS_Data/10_diffShapesBEST/Labels_DataSet.npy')

    # ds_poly = GP.SlighlyMoreClevr(n_gons=n_gons, canvas_size=canvas_size, size_of_ds_poly=size_of_ds_poly)  # Generate dataset
    # random_indices_poly = torch.randperm(len(ds_poly))
    random_indices_poly = torch.randperm(len(Inputs))
    # Class for loading Polygons sequence from a sequence folder
    generator = GP.DenoiseHPatchesPoly_Exp6(random_indices_poly=random_indices_poly, inputs=Inputs,labels=Labels, batch_size=50)

    # generator= GP.DenoiseHPatchesPoly_Exp5(random_indices_poly=random_indices_poly, inputs=Inputs,
    #                                                      labels=Labels, batch_size=50)
    generator_for_5x6 = GP.DenoiseHPatchesPoly_Stage_1_3(random_indices_poly=random_indices_poly,
                                                              inputs=Inputs, labels=Labels, batch_size=50)

    # generator = GP.DenoiseHPatchesPoly_Stage_0(random_indices_poly=random_indices_poly, ds_poly=Inputs, batch_size=50)
    # generator = GP.DenoiseHPatchesPoly(random_indices_poly=random_indices_poly, ds_poly=ds_poly, batch_size=50)
    # generator = GP.DenoiseHPatchesPoly_Exp4(random_indices_poly=random_indices_poly, ds_poly=ds_poly, batch_size=50)

    for index in range(1):
        # index = 4
        imgs = generator[3][0]
        imgs2 = generator_for_5x6[3][0]
        imgs_input = generator[3][0][index,:,:,0]
        imgs_clean = generator[3][1][index,:,:,0]
        # index = np.random.randint(0, imgs.shape[0])
        # StressToStress = generator[3][1]
        Shallow_model_prediction = Shallow_model.predict(imgs)
        Baseline_model_prediction = Baseline_model.predict(imgs)
        Full_model_prediction = Full_model.predict(imgs)
        Half_model_prediction = Half_model.predict(imgs2)


        # ===============================================
        # # plt.subplot(151)
        # # plt.imshow(imgs[index,:,:,0], cmap='gray')
        # # plt.title('Input', fontsize=15)
        # # plt.gca().set_xticks([])
        # # plt.gca().set_yticks([])
        #
        # plt.subplot(141)
        # plt.imshow(imgs_clean[index, :, :, 0], cmap='jet')
        # plt.title('Target', fontsize=15)
        # plt.gca().set_xticks([])
        # plt.gca().set_yticks([])
        #
        # plt.subplot(142)
        # plt.imshow(Baseline_model_prediction[index,:,:,0], cmap='jet')
        # plt.title('Minimal', fontsize=15)
        # plt.gca().set_xticks([])
        # plt.gca().set_yticks([])
        #
        # plt.subplot(143)
        # plt.imshow(Full_model_prediction[index, :, :, 0], cmap='jet')
        # plt.title('Full', fontsize=15)
        # plt.gca().set_xticks([])
        # plt.gca().set_yticks([])
        #
        # plt.subplot(144)
        # plt.imshow(Shallow_model_prediction[index, :, :, 0], cmap='jet')
        # plt.title('Shallow', fontsize=15)
        # plt.gca().set_xticks([])
        # plt.gca().set_yticks([])
        #
        # plt.colorbar()
        # # plt.show()
        # ===============================================

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, gridspec_kw={'width_ratios': [1.37, 1, 1, 1, 1]})

        # plot just the positive data and save the
        # color "mappable" object returned by ax1.imshow
        target_im = ax1.imshow(imgs_clean, cmap='jet', interpolation='none')
        ax1.title.set_text('Target')
        ax1.axis('off')
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("left", size="5%", pad=0.38)

        cb = plt.colorbar(target_im, cax=cax)
        ax1.yaxis.set_ticks_position('left')

        baseline_im = ax2.imshow(Baseline_model_prediction[index,:,:,0], cmap='jet', interpolation='none')
        ax2.title.set_text('Minimal')
        ax2.axis('off')

        full_im = ax3.imshow(Full_model_prediction[index, :, :, 0], cmap='jet', interpolation='none')
        ax3.title.set_text('Full')
        ax3.axis('off')

        shallow_im = ax4.imshow(Shallow_model_prediction[index, :, :, 0], cmap='jet', interpolation='none')
        ax4.title.set_text('Shallow')
        ax4.axis('off')

        half_im = ax5.imshow(Half_model_prediction[index,:,:,0], cmap='jet', interpolation='none')
        ax5.title.set_text('Half')
        ax5.axis('off')

        plt.show()


        # plt.savefig('./Results/' + str(index) + '.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    plot_sample()