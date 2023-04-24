import cv2 as cv
from matplotlib import pyplot as plt

my_img = cv.imread("./images/cat.jpg", cv.IMREAD_COLOR )

# print(my_img)
# print(type(my_img))
# print(my_img.shape)

# cv.imshow("My Cat", my_img)
# cv.waitKey(0)

my_img_hsv = cv.cvtColor(my_img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(my_img_hsv)
plt.figure(figsize=[12,4])
plt.subplot(151)
plt.imshow(my_img[:,:,::-1])
plt.subplot(152)
plt.imshow(h, cmap='gray')
plt.subplot(153)
plt.imshow(s, cmap='gray')
plt.subplot(154)
plt.imshow(v, cmap='gray')

my_img_merged = cv.cvtColor(cv.merge((h+10, s, v)), cv.COLOR_HSV2RGB)
plt.subplot(155)
plt.imshow(my_img_merged)
plt.show()

