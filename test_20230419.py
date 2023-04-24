import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

my_img = cv.imread("./images/cat.jpg", cv.IMREAD_COLOR)
my_background = cv.imread("./images/checkerboard_color.png", cv.IMREAD_COLOR)

logo_w = my_img.shape[1]
logo_h = my_img.shape[0]
#aspect_ratio = logo_w / my_background.shape[1]
#dim = (logo_w, int(my_background.shape[0] * aspect_ratio))
dim = (logo_w, logo_h)
img_background = cv.resize(my_background, dim, interpolation=cv.INTER_AREA)

my_img_gray = cv.cvtColor(my_img, cv.COLOR_BGRA2GRAY)
retval, img_thresh = cv.threshold(my_img_gray, 100, 255, cv.THRESH_BINARY)
img_mask_inv = cv.bitwise_not(img_thresh)

img_background_mask = cv.bitwise_and(img_background, img_background, mask=img_thresh)
img_foreground = cv.bitwise_and(my_img, my_img, mask=img_mask_inv)

result = cv.add(img_foreground, img_background_mask)

plt.figure(figsize=[10,5])
plt.subplot(121); 
plt.imshow(my_img[:,:,::-1]);         
plt.title("Foreground");

plt.subplot(122); 
plt.imshow(result[:,:,::-1]);       
plt.title("Background");

plt.show()

#---------------------

# matrix_1 = np.ones(my_img.shape, dtype='uint8') * 0.5
# matrix_2 = np.ones(my_img.shape, dtype='uint8') * 1.5

# lower_contrast = np.uint8(cv.multiply(np.float64(my_img), matrix_1))
# higher_contrast = np.uint8(np.clip(cv.multiply(np.float64(my_img), matrix_2), 0, 255))

# plt.figure(figsize=(12, 5))

# plt.suptitle("Arithmetic Operations")

# plt.subplot(131)
# plt.imshow(lower_contrast)
# plt.title("Lower Contrast")

# plt.subplot(132)
# plt.imshow(my_img)
# plt.title("Original")

# plt.subplot(133)
# plt.imshow(higher_contrast)
# plt.title("Higher Contrast")

# plt.show()


# matrix = np.ones(my_img.shape, dtype='uint8')*75

# britghter_img = cv.add(my_img, matrix)
# darker_img = cv.subtract(my_img, matrix)

# plt.figure(figsize=(12, 5))

# plt.suptitle("Arithmetic Operations")

# plt.subplot(131)
# plt.imshow(darker_img)
# plt.title("Darker")

# plt.subplot(132)
# plt.imshow(my_img)
# plt.title("Original")

# plt.subplot(133)
# plt.imshow(britghter_img)
# plt.title("Brightier")

# plt.show()