import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

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


if __name__ == '__main__':
    img = cv2.imread('./data/Train/Images/a0.jpg', cv2.IMREAD_UNCHANGED)
    mask = cv2.imread('./data/Train/Masks/a0.jpg', cv2.IMREAD_UNCHANGED)
    img1 = Image.open('./data/Train/Images/a0.jpg')
    img1 = np.asarray(img1)
    mask1 = Image.open('./data/Train/Masks/a0.jpg')

    print(dir(img1))
    print(img.shape)
    print(img1.shape)
    print(mask.shape)
    print(mask1.size)
    print(type(img), type(img1))

    print(img[1][2])
    # plot_img_and_mask(img, mask)
    

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(img)

    ax[0, 1].set_title('Output mask')
    ax[0, 1].imshow(mask)

    ax[1, 0].imshow(img1)

    ax[1, 1].set_title('Output mask')
    ax[1, 1].imshow(mask1)
    plt.show()

    # print(dir(img))
    # print(img.size)