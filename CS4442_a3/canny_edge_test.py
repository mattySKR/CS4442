# Student: Matvey Skripchenko
# Student number: 250899673

import canny_edge_class as canny
import matplotlib.pyplot as plt

# Will be used to put everything together
img = []

# Extracting our image
im = plt.imread('carski.jpeg')

# Even though we are already given the greyscale image
# for the assignment, I have still decided to implement
# this part so that any image can be used
r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

img.append(img_gray)

# Creating our cranny edge object with the specified values
# CAN CHANGE THE VALUES HERE IF YOU LIKE
detector_obj = canny.cannyEdge(img, sigma=1.4, low_threshold=0.09, high_threshold=0.17)

# Obtaining the final image
img_finale = detector_obj.detector()


for i, im in enumerate(img_finale):
    if im.shape[0] == 3:
        im = im.transpose(1,2,0)

    plt.imshow(im, cmap='gray')

plt.show()

