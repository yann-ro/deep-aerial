import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_seg_mask(mask, ax, title, labels):
    im = ax.imshow(mask)
    ax.set_title(title)
    ax.set_axis_off()

    label = labels.loc[mask.unique().tolist()].name.to_list()
    values = np.unique(mask.ravel())
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [
        mpatches.Patch(color=colors[i], label=lab) for i, lab in enumerate(label)
    ]

    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


def display_results(i, test_set, segmodel, label):
    image1, mask1 = test_set[i]
    pred_mask1, score1 = segmodel.predict_image_mask_miou(image1, mask1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(image1)
    ax1.set_title("Picture")

    plot_seg_mask(mask1, ax2, "Ground truth", label)
    plot_seg_mask(pred_mask1, ax3, "UNet-MobileNet | mIoU {:.3f}".format(score1), label)

    fig.tight_layout()
