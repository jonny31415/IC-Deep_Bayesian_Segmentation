import time
import numpy as np
import tensorflow as tf
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import matplotlib.colors as colors

DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
# DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
DEFAULT_CMAP = 'hot'
DEFAULT_NORM = colors.Normalize(vmin=0, vmax=1,)
DEFAULT_N_GRID = 100

def print_np(var):
    print(var.shape, var.dtype, np.min(var), np.max(var))

def PIL2numpy(img_pil):
    img_np = np.array(img_pil)
    
    if len(img_np.shape)==3:
        #img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        pass

    return img_np

def get_layers(model):
    name = model.name
    file = f'./{name}'+'-layers.txt'

    with open(file, 'w') as f:
        try:
            i = 0
            while(1):
                f.write(str(model.get_layer(index=i).get_config()) + '\n')
                i+=1
        except Exception as e:
            print(e)


def get_summary(model):
    name = model.name
    file = f'./{name}'+'-summary.txt'

    with open(file, 'w') as f:
        with redirect_stdout(f):
            model.summary()

def pause_for(sec):
    print("Paused for {}s...".format(sec))
    time.sleep(sec)

# Visualize uncertainties

def plot_uncertainty_surface(test_uncertainty, ax, shape, cmap=None):
    """Visualizes the 2D uncertainty surface.

    For simplicity, assume these objects already exist in the memory:

        test_examples: Array of test examples, shape (num_test, 2).
        train_labels: Array of train labels, shape (num_train, ).
        train_examples: Array of train examples, shape (num_train, 2).

    Arguments:
        test_uncertainty: Array of uncertainty scores, shape (num_test,).
        ax: A matplotlib Axes object that specifies a matplotlib figure.
        cmap: A matplotlib colormap object specifying the palette of the
        predictive surface.

    Returns:
        pcm: A matplotlib PathCollection object that contains the palette
        information of the uncertainty plot.
    """
    # Normalize uncertainty for better visualization.
    test_uncertainty = test_uncertainty / np.max(test_uncertainty)

    # Set view limits.
    ax.set_ylim(reversed(DEFAULT_Y_RANGE))
    ax.set_xlim(DEFAULT_X_RANGE)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    # Plot normalized uncertainty surface.
    pcm = ax.imshow(
        np.reshape(test_uncertainty, shape),
        cmap=cmap,
        origin="lower",
        extent=DEFAULT_X_RANGE + DEFAULT_Y_RANGE,
        vmin=DEFAULT_NORM.vmin,
        vmax=DEFAULT_NORM.vmax,
        interpolation='bicubic',
        aspect='auto')

    # # Plot training data.
    # ax.scatter(train_examples[:, 0], train_examples[:, 1],
    #             c=train_labels, cmap=DEFAULT_CMAP, alpha=0.5)
    # ax.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

    return pcm

def plot_predictions(variance, model_name=""):
    """Plot normalized class probabilities and predictive uncertainties."""
    # Compute predictive uncertainty.

    H, W = variance.shape[2], variance.shape[3]

    # Initialize the plot axes.
    fig, axs = plt.subplots(1, 1, figsize=(14, 5))

    uncertaity_reduced = tf.math.reduce_sum(variance, axis=-1)

    # Plots the predictive uncertainty.
    pcm_0 = plot_uncertainty_surface(uncertaity_reduced[0,0,:,:], ax=axs, shape=(H, W), cmap='hot')

    # Adds color bars and titles.
    fig.colorbar(pcm_0, ax=axs)

    axs.set_title(f"(Normalized) Predictive Uncertainty {model_name}")

    plt.savefig("Uncertainty.png")