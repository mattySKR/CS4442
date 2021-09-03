# Student: Matvey Skripchenko
# Student number: 250899673
import cv2
import numpy as np


# This method will find the corners and throw them on the input image
# k here is the Harris corneer constant, which should usually be
# between 0.04 and 0.06. The rest of the variables are pretty straightforward
# from their name.
# The image with identified corners will be returned.

def harris_corner(img, window_size, harr_const, threshold):

    # Calculating x and y derivative
    dy, dx = np.gradient(img)
    i_xx = dx**2
    i_xy = dy*dx
    i_yy = dy**2
    height = img.shape[0]
    width = img.shape[1]

    new_im = img.copy()
    c_img = cv2.cvtColor(new_im, cv2.COLOR_GRAY2RGB)
    offset = window_size // 2

    # Here, we are looping through the image and finding the corners
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):

            # Here will calculate the sum of the squares
            window_i_xx = i_xx[y-offset:y+offset+1, x-offset:x+offset+1]
            window_i_xy = i_xy[y-offset:y+offset+1, x-offset:x+offset+1]
            window_i_yy = i_yy[y-offset:y+offset+1, x-offset:x+offset+1]

            s_xx = window_i_xx.sum()
            s_xy = window_i_xy.sum()
            s_yy = window_i_yy.sum()

            # Now, finding the determinant and the trace in order to
            # obtain the corner responses
            det = (s_xx * s_yy) - (s_xy**2)
            trace = s_xx + s_yy
            r = det - harr_const*(trace**2)

            # Finally, if the corner response is higher than the given threshold, then
            # the point will be coloured
            if r > threshold:

                c_img.itemset((y, x, 0), 0)
                c_img.itemset((y, x, 1), 0)
                c_img.itemset((y, x, 2), 255)

    return c_img


# Main
def main():

    # Here you can choose whatever picture you like,
    # but first make sure it is in the current
    # working directory
    img = cv2.imread('house.jfif')

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if len(img.shape) == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # Here, can play around with different values that
    # the algorithm takes
    final_image = harris_corner(img, 3, 0.06, 1000000)

    # Saving the image to the working directory
    cv2.imwrite("Harris_Corner.png", final_image)

if __name__ == "__main__":
    main()
