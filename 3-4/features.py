import cv2
import mahotas
import numpy as np
bins = 8

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def fd_4(image, mask=None):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(1,1),0)
    edged = cv2.Canny(gray,40,250)
    cv2.normalize(edged,gray)
    return edged.flatten()

def fd_5(image, mask=None):
    figure_size = 9
    new_image = cv2.medianBlur(image, figure_size)
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray,40,250)
    cv2.normalize(edged,gray)
    return edged.flatten()

def fd_6(image, mask=None):
    imag = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    imag=cv2.drawKeypoints(gray,kp,imag)
    cv2.normalize(imag, imag)
    return imag.flatten()