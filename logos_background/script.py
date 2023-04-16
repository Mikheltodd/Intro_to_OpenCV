from turtle import back
import cv2 as cv
import numpy as np

def rescaleImage(image, w_scale=0.75, h_scale=0.75):
    width = int(image.shape[1]*w_scale)
    height = int(image.shape[0]*h_scale)
    dim = (width, height)
    return cv.resize(image, dim, interpolation=cv.INTER_AREA)

def mergeImages(foreground, background):
    foreground_dim = foreground.shape
    background_dim = background.shape

    # if foreground_dim[1] < background_dim[1]:
    #     foreground_rescale = foreground
    #     w_scale = foreground.shape[1]/background.shape[1]
    #     h_scale = foreground.shape[0]/background.shape[0]
    #     background_rescale = rescaleImage(background, w_scale, h_scale)
    # else:
    #     background_rescale = background
    #     w_scale = background.shape[1]/foreground.shape[1]
    #     h_scale = background.shape[0]/foreground.shape[0]
    #     foreground_rescale = rescaleImage(foreground, w_scale, h_scale)

    foreground_rescale = foreground
    w_scale = foreground.shape[1]/background.shape[1]
    h_scale = foreground.shape[0]/background.shape[0]
    background_rescale = rescaleImage(background, w_scale, h_scale)

    foreground_gray = cv.cvtColor(foreground_rescale, cv.COLOR_RGB2GRAY)

    retval, foreground_mask = cv.threshold(foreground_gray, 100, 255, cv.THRESH_BINARY)

    foreground_mask_inv = cv.bitwise_not(foreground_mask)

    background_image = cv.bitwise_and(background_rescale, background_rescale, mask=foreground_mask)
    foreground_image = cv.bitwise_and(foreground_rescale, foreground_rescale, mask=foreground_mask_inv)

    new_image = cv.add(background_image, foreground_image)

    cv.imshow('Foreground', foreground_rescale)
    cv.imshow('Background', background_rescale)
    cv.imshow('Foreground Gray', foreground_gray)
    cv.imshow('Foreground Mask', foreground_mask)
    cv.imshow('Foreground Mask Inverted', foreground_mask_inv)
    cv.imshow('Foreground Image', foreground_image)
    cv.imshow('Background Image', background_image)
    # cv.imshow('New Image',new_image)

    return new_image







foreground = cv.imread('logos/pepsi.png')
background = cv.imread('photos/ice.jpeg')
new_image = mergeImages(foreground, background)

cv.imshow('New Image', new_image)

cv.waitKey(0)