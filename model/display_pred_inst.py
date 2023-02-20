import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, patheffects


def add_bbox(ax, bbox, conf=None, color="red"):
    bbox = np.array(bbox.cpu().detach(), dtype=np.float32)
    ax.add_patch(
        patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            color=color,
            fill=False,
            lw=1,
        )
    )
    if conf:
        ax.text(
            bbox[0] + 2,
            (bbox[1] - 7),
            f"{float(conf):.3f}",
            verticalalignment="top",
            color="white",
            fontsize=10,
            weight="bold",
        ).set_path_effects(
            [patheffects.Stroke(linewidth=2, foreground="black"), patheffects.Normal()]
        )


def display_img_with_bbox(ax, img, labels, legend=None, from_pred=False):
    ax.imshow(img)
    ax.axis("off")

    if from_pred:
        for bbox, label, score in zip(
            labels["boxes"], labels["labels"], labels["scores"]
        ):
            add_bbox(ax, bbox, score)
    else:
        for bbox, label in zip(labels["boxes"], labels["labels"]):
            add_bbox(ax, bbox)


def display_output_inst(model, test_set, device, list_id):
    model.eval()
    fig, axs = plt.subplots(1, len(list_id), figsize=(20, 10))

    for i, id in enumerate(list_id):
        img, _ = test_set[id]
        (pred,) = model(img[None, :].to(device))

        display_img_with_bbox(axs[i], img.permute(1, 2, 0).cpu(), pred, from_pred=True)
    fig.show()
