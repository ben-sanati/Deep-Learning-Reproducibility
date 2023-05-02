from qbstyles import mpl_style

def _style_ticks(axis, color):
    """ Enable minor ticks, and color major + minor ticks"""
    axis.minorticks_on()
    ticks = (
        axis.get_xticklines()
        + axis.xaxis.get_minorticklines()
        + axis.get_yticklines()
        + axis.yaxis.get_minorticklines()
    )

    for tick in ticks:
        tick.set_color("#" + color + "3D")

def _monkey_patch_subplot(color, subplot):
    """ Style all axes of a figure containing subplots, just after the
    figure is created. """

    def _patch(*args, **kwargs):
        fig, axes = subplot(*args, **kwargs)
        axes_list = [axes] if isinstance(axes, matplotlib.axes.Axes) else axes
        for ax in axes_list:
            if isinstance(ax, matplotlib.axes.Axes):
                _style_ticks(ax, color)
            else:
                for each in ax:
                    _style_ticks(each, color)
        return fig, axes

    return _patch

def plot(dark):
    mpl_style(dark, minor_ticks=False)
    plt.subplots = _monkey_patch_subplot("FFFFFF", plt.subplots)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for pos in range(len(Epochs)):
        axes[0, 0].plot(Epochs[pos][1:], LossTr[pos][1:], label=f"Model {(pos+1)*100000}")
        axes[0, 0].scatter(Epochs[pos][1:], LossTr[pos][1:])

        axes[0, 1].plot(Epochs[pos], LossVa[pos], label=f"Model {(pos+1)*100000}")
        axes[0, 1].scatter(Epochs[pos], LossVa[pos])

        axes[1, 0].plot(Epochs[pos], AccuTr[pos], label=f"Model {(pos+1)*100000}")
        axes[1, 0].scatter(Epochs[pos], AccuTr[pos])

        axes[1, 1].plot(Epochs[pos], AccuVa[pos], label=f"Model {(pos+1)*100000}")
        axes[1, 1].scatter(Epochs[pos], AccuVa[pos])

    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[0, 0].set_title('Training Loss Plots')
    axes[0, 1].set_title('Validation Loss Plots')
    axes[1, 0].set_title('Training Accuracy Plots')
    axes[1, 1].set_title('Validation Accuracy Plots')
    fig.suptitle('Exercise 4 Empirical Overfitting Plots', fontsize=16)
    plt.savefig('plots.png')
