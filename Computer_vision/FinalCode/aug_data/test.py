import matplotlib.pyplot as plt
import dlib
import numpy as np
import skimage
from PIL import Image

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

img = dlib.load_rgb_image('../jaffe/ANGRY/angry4.jpg')

rect = detector(img)[0]
sp = predictor(img, rect)
landmarks = np.array([[p.x, p.y] for p in sp.parts()])
outline = landmarks[[*range(17), *range(26, 16, -1)]]

Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])

cropped_img = np.zeros(img.shape, dtype=np.uint8)
cropped_img[Y, X] = img[Y, X]

img_mask = np.zeros(img.shape, dtype=np.uint8)
img_mask[Y, X] = 1

plt.imshow(img_mask[:,:,2])

img_mask.shape

Image.open(img)
# plt.imshow(img)
# plt.imshow(cropped_img)
