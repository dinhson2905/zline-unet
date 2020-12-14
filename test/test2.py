from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

pil_img = Image.open('data/Train/Masks/a0.jpg').convert('L')

# pil_img = Image.open('data/Train/Images/a0.jpg')

w, h = pil_img.size
newW, newH = int(1 * w), int(1 * h)
assert newW > 0 and newH > 0, 'Scale is too small'
pil_img = pil_img.resize((newW, newH))

img_nd = np.array(pil_img)
print("image: ", img_nd.shape)
print(img_nd)
if len(img_nd.shape) == 2:
    img_nd = np.expand_dims(img_nd, axis=2)

img_trans = img_nd.transpose((2, 0, 1))
if img_trans.max() > 1:
    img_trans = img_trans / 255

print("trans: ", img_trans.shape)
print(img_trans)

res = img_trans == 1
k = res[True]
print(k)