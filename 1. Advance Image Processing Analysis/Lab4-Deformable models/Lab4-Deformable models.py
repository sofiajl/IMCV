#!/usr/bin/env python3

'''
AIPA Lab 4 - Deformable models and level sets

Usage:
  python3 lab4-code.py [<image> <outpath>]

Keys:
  r     - reset the marks
  w     - Watershed segmentation
  c     - Chan-Vese segmentation
  a     - Active contours
  m     - Morphological active contours
  g     - Morphological geodesic active contours
  ESC   - exit
'''

import sys
import os
import argparse
import numpy as np

import cv2 as cv
from skimage import segmentation, measure, color, img_as_float, filters, exposure, morphology
from scipy import ndimage


class Marker:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        
        self.show()
        cv.moveWindow(self.windowname, 0, 0)
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv.line(dst, self.prev_pt, pt, color, 10)
            self.dirty = True
            self.prev_pt = pt
            self.show()

def watershed_segmentation(img, **kwargs):
    if img.ndim > 2:
        img = color.rgb2gray(img)
    labels = measure.label(kwargs['mark'])
    gradient_img = - ndimage.gaussian_gradient_magnitude(img, kwargs['sigma'])
    output = segmentation.watershed(gradient_img, labels)
    output = output.astype(np.uint8) * int(255 / len(np.unique(output)))
    return output


def chan_vese_segmentation(img, **kwargs):
    if img.ndim > 2:
        img = color.rgb2gray(img)
        
    #output = segmentation.chan_vese(img, init_level_set="checkerboard", extended_output=False)
    output = segmentation.chan_vese(img, init_level_set=kwargs['mark'], extended_output=False)
    output = (output * 255).astype(np.uint8)
    return output

def active_contours_segmentation(img, **kwargs):
    if img.ndim > 2:
        img = color.rgb2gray(img)

    s = np.linspace(0, 4*np.pi, 800)
    x = 220 + 100*np.cos(s)
    y = 100 + 100*np.sin(s)
    init = np.array([x, y]).T
    snake = segmentation.active_contour(filters.gaussian(img, 3), init, alpha=0.1, beta=10, gamma=0.01, max_iterations=kwargs['max_it'])
    
    output = np.zeros(img.shape)
    output[np.int_(snake[:,0]), np.int_(snake[:,1])] = 255
    
    return output

def morph_active_contours_segmentation(img, **kwargs):
    if img.ndim > 2:
        img = color.rgb2gray(img)
        
    def store_evolution_in(lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """

        def _store(x):
            lst.append(np.copy(x))

        return _store

    # Morphological ACWE
    image = img_as_float(img)

    # Histogram equalization
    img = exposure.equalize_hist(img)

    # Median filtering
    img = filters.median(img, selem=morphology.disk(2))
    
    #Morphological Chan-Vese
    output = segmentation.morphological_chan_vese(image, iterations=80, init_level_set=kwargs['mark'],
                                 smoothing=1)
    output = (output * 255).astype(np.uint8)
    return output


def morph_geo_active_contours_segmentation(img, **kwargs):
    if img.ndim > 2:
        img = color.rgb2gray(img)

    #image = img_as_float(img)
    # Histogram equalization
    img = exposure.equalize_hist(img)

    # Median filtering
    img = filters.median(img, selem=morphology.disk(2))
    img = np.max(img) - filters.gaussian(img, sigma=kwargs['sigma'])
     
    # Applying Morphological Geodesic Active Contour segmentation with specific parameters
    im_morpho = segmentation.morphological_geodesic_active_contour(img, iterations=40,
                                                     init_level_set=kwargs['mark'],
                                                     smoothing=2)
    output = (im_morpho * 255).astype(np.uint8)
    return output


# Write your code for segmentation methods here

if __name__ == '__main__':
    print(__doc__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='images/mitochondria.jpg') 
    parser.add_argument('--outpath', type=str, default='output') 
    args = parser.parse_args()

    img = cv.imread(args.image)
    name = os.path.splitext(os.path.basename(args.image))[0]
    
    if img is None:
        print('Failed to load image file:', args.image)
        sys.exit(1)

    img_mark = img.copy()
    mark = np.zeros(img.shape[:2], np.uint8)
    marker = Marker('img', [img_mark, mark], lambda : ((0, 0, 255), 255))

    output = None
    method = ''

    while True:
        ch = cv.waitKey()
        if ch == 27:
            break
        if ch == ord('r'):
            img_mark[:] = img
            mark[:] = 0
            marker.show()
        if ch == ord('w'):
            method = "watershed_segmentation"
            print(f"Applying {method} method")
            output = watershed_segmentation(img, mark=mark, sigma=4)
        if ch == ord('c'):
            method = "chan_vese_segmentation"
            print(f"Applying {method} method")
            output = chan_vese_segmentation(img, mark=mark)
        if ch == ord('a'):
            method = "active_contours_segmentation"
            output = active_contours_segmentation(img, max_it=100)
            print(f"Method to be implemented ({method})")
        if ch == ord('m'):
            method = "morph_active_contours_segmentation"
            output = morph_active_contours_segmentation(img, mark=mark)
            print(f"Method to be implemented ({method})")
        if ch == ord('g'):
            method = "morph_geo_active_contours_segmentation"
            output = morph_geo_active_contours_segmentation(img, mark=mark, sigma=4.0)
            print(f"Method to be implemented ({method})")
        if output is not None:
            cv.namedWindow('output')        
            cv.moveWindow('output', img.shape[1], 0)
            cv.imshow('output', output)
        if ch == ord('s'):
            if output is not None:
                outfile = f"{args.outpath}/{name}_{method}.jpg"
                print(f'Saving {outfile}')
                cv.imwrite(outfile, output)
            else:
                print(f"No output to be saved")


    cv.destroyAllWindows()
    print('Done')
