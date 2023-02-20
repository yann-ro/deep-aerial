import matplotlib.pyplot as plt
import numpy as np

def create_rect(bbox, color='red'):
    
    bbox = np.array(bbox.cpu().detach(), dtype=np.float32)
    return plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2]-bbox[0],
            bbox[3]-bbox[1],
            color = color,
            fill = False,
            lw = 1
        )

def display_img_with_bbox(im, labels, legend=None):
    
    plt.figure(figsize=(7,7))
    plt.imshow(im)
    plt.axis('off')
    for bbox, label in zip(labels["boxes"], labels["labels"]):
        plt.gca().add_patch(create_rect(bbox))