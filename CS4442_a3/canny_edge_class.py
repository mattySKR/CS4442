# Student: Matvey Skripchenko
# Student number: 250899673

from scipy import ndimage
from scipy.ndimage.filters import convolve
import numpy as np


class cannyEdge:

    # Declaring all the needed variable for the cannyEdge object
    def __init__(self, img, sigma, low_threshold, high_threshold, kernel_dim=3, weak_pix=100, strong_pix=255):

        self.img = img
        self.img_finale = []
        self.img_smoothed_out = None
        self.gradient_matrix = None
        self.angle_matrix = None
        self.non_max_sup_img = None
        self.threshold_img = None
        self.weak_pix = weak_pix
        self.strong_pix = strong_pix
        self.sigma = sigma
        self.kernel_dim = kernel_dim
        self.low_t = low_threshold
        self.high_t = high_threshold
        return

    # Gaussian kernel method declaration
    # Will be used for noise reduction and I will choose
    # the kernel size to be 3x3 dimension.
    # Important to note that the smaller the kernel size, the less
    # visible the blur(noise) will be. The smallest we can go is 3x3.
    def gaussian(self, size, sigma=1):

        size = int(size) // 2    # maybe change to int(size)
        x, y = np.mgrid[-size:size+1, -size:size+1]
        norm = 1 / (2.0 * np.pi * sigma**2)
        gauss_kernel = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * norm
        return gauss_kernel

    # Method for gradient calculation: gradient magnitude and
    # slope of the gradient. This will be used for detecting the
    # edge intensity and its direction. So, we apply Sobel filters
    # that will highlight the intensity change in x and y directions.
    # So, we convolve the image with Sobel edge detection filter
    def sobels(self, im):

        k_x_dir = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        k_y_dir = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        i_x = ndimage.filters.convolve(im, k_x_dir)
        i_y = ndimage.filters.convolve(im, k_y_dir)

        g_val = np.hypot(i_x, i_y)
        g_val = g_val / g_val.max() * 255
        theta_angle = np.arctan2(i_y, i_x)

        return g_val, theta_angle


    # Non max suppression method will thin out the edges.
    # Thus, it needs to go through all gradient intensity
    # matrix point and pick the pixels the max values in
    # the directions of the edge at scope. This is done
    # for the purpose of achieving the same intensity of
    # the edges.
    def non_max_suppress(self, im, D):

        M, N = im.shape
        processed_img = np.zeros((M,N), dtype=np.int32)
        ang = D * 180. / np.pi                  # might need 180. /
        ang[ang < 0] += 180

        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    q_val = 255
                    r_val = 255

                    # 0 degree angle
                    if (0 <= ang[i,j] < 22.5) or (157.5 <= ang[i,j] <= 180):
                        q_val = im[i, j+1]
                        r_val = im[i, j-1]

                    # 45 degree angle
                    elif 22.5 <= ang[i,j] < 67.5:
                        q_val = im[i+1, j-1]
                        r_val = im[i-1, j+1]

                    # 90 degree angle
                    elif 67.5 <= ang[i,j] < 112.5:
                        q_val = im[i+1, j]
                        r_val = im[i-1, j]

                    # 135 degree angle
                    elif 112.5 <= ang[i,j] < 157.5:
                        q_val = im[i-1, j-1]
                        r_val = im[i+1, j+1]

                    # final step
                    if (im[i,j] >= q_val) and (im[i,j] >= r_val):
                        processed_img[i,j] = im[i,j]
                    else:
                        processed_img[i,j] = 0

                except IndexError as e:
                    pass

        return processed_img

    # This method will be used for identifying the weak,
    # strong and irrelevant pixels. Thus, only weak and strong
    # pixels are of interest here (the ones that will be included).
    #
    # If the intensity of the pixel is lower than the given low
    # threshold, then it is irrelevant.
    # If the intensity of the pixel is higher than the given high
    # threshold, then it will be the strong pixel.
    # If the intensity of the pixel is in the interval between the
    # given low and high thresholds, then it will be the weak pixel.
    # But, for better identifying the weak ones, hysteresis will later
    # be used.
    def threshold(self, im):

        high_t = im.max() * self.high_t
        low_t = high_t * self.low_t

        M, N = im.shape
        final = np.zeros((M,N), dtype=np.int32)

        strong_val = np.int32(self.strong_pix)
        weak_val = np.int32(self.weak_pix)


        # Collecting the values for appropriate pixels
        strong_pix_i, strong_pix_j = np.where(im >= high_t)
        weak_pix_i, weak_pix_j = np.where((im <= high_t) & (im >= low_t))

        irrelev_i, irrelev_j = np.where(im < low_t) # technically do not need this


        final[strong_pix_i, strong_pix_j] = strong_val
        final[weak_pix_i, weak_pix_j] = weak_val

        return final

    # As mentioned previously, this method will be used
    # for transforming weak pixels into strong ones.
    # This is achieved by checking whether there is
    # at least one of pixels around the one at scope is
    # a strong pixel.
    def hysteresis(self, im):

        M, N = im.shape
        weak_val = self.weak_pix
        strong_val = self.strong_pix

        for i in range(1, M-1):
            for j in range(1, N-1):

                if im[i,j] == weak_val:
                    try:

                        if (im[i+1, j-1] == strong_val) or (im[i+1, j] == strong_val) or (im[i+1, j+1] == strong_val) or (im[i, j-1] == strong_val) or (im[i, j+1] == strong_val) or (im[i-1, j-1] == strong_val) or (im[i-1, j] == strong_val) or (im[i-1, j+1] == strong_val):
                            im[i, j] = strong_val
                        else:
                            im[i, j] = 0

                    except IndexError as e:
                        pass

        return im

    # Collecting and putting everything together to
    # get the required binary image
    def detector(self):

        for i, im in enumerate(self.img):

            self.img_smoothed_out = convolve(im, self.gaussian(self.kernel_dim, self.sigma))
            self.gradient_matrix, self.angle_matrix = self.sobels(self.img_smoothed_out)
            self.non_max_sup_img = self.non_max_suppress(self.gradient_matrix, self.angle_matrix)
            self.threshold_img = self.threshold(self.non_max_sup_img)
            final_image = self.hysteresis(self.threshold_img)
            self.img_finale.append(final_image)

        return self.img_finale



