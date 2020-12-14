import matplotlib.pyplot as plt
from PIL import Image

def plot_img_and_mask(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)

    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title('Output mask')
        ax[1].imshow(mask)
    
    plt.xticks([]), plt.yticks([])
    plt.show()


# if __name__ == '__main__':
#     img = Image.open('data/Val/Images/a187.jpg')
#     mask = Image.open('data/Val/Masks/a187.jpg')
#     plot_img_and_mask(img, mask)