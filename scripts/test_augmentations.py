from augmentations import augment
import cv2
import numpy as np

def merge(img_arr):
    num_images = len(img_arr)
    num_rows = np.math.ceil(np.math.sqrt(num_images))
    num_cols = np.math.ceil(num_images / num_rows)
    total_image = None
    for row_num in range(num_rows):
        idx = row_num * num_cols
        image = img_arr[idx]
        for img in img_arr[idx + 1: idx + num_cols]:
            image = np.hstack((image, img))
        if total_image is None:
            total_image = image
        else:
            total_image = np.vstack((total_image, image))

    return total_image

image = cv2.imread('Corn_dataset/images/770.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

images = augment([image for x in range(30)])

image = merge(images)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cv2.imwrite('augmented.png', image)