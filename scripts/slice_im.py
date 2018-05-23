# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 02:53:01 2016

@author: avanetten
"""

import os
import cv2
import time

###############################################################################    
def slice_im(image_path, out_name, outdir, sliceHeight=256, sliceWidth=256, 
             zero_frac_thresh=0.2, overlap=0.2, verbose=False):
    '''Slice large satellite image into smaller pieces, 
    ignore slices with a percentage null greater then zero_fract_thresh
    Assume three bands!'''

    image0 = cv2.imread(image_path, 1)  # color
    ext = '.' + image_path.split('.')[-1]
    win_h, win_w = image0.shape[:2]
    
    # if slice sizes are large than image, pad the edges
    pad = 0
    if sliceHeight > win_h:
        pad = sliceHeight - win_h
    if sliceWidth > win_w:
        pad = max(pad, sliceWidth - win_w)
    # pad the edge of the image with black pixels
    if pad > 0:    
        border_color = (0,0,0)
        image0 = cv2.copyMakeBorder(image0, pad, pad, pad, pad, 
                                 cv2.BORDER_CONSTANT, value=border_color)

    win_size = sliceHeight*sliceWidth

    t0 = time.time()
    n_ims = 0
    n_ims_nonull = 0
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    for y0 in xrange(0, image0.shape[0], dy):#sliceHeight):
        for x0 in xrange(0, image0.shape[1], dx):#sliceWidth):
            n_ims += 1
            
            # make sure we don't have a tiny image on the edge
            if y0+sliceHeight > image0.shape[0]:
                y = image0.shape[0] - sliceHeight
            else:
                y = y0
            if x0+sliceWidth > image0.shape[1]:
                x = image0.shape[1] - sliceWidth
            else:
                x = x0
        
            # extract image
            window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
            # get black and white image
            window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)
            
            # find threshold that's not black
            # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html?highlight=threshold
            ret,thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
            non_zero_counts = cv2.countNonZero(thresh1)
            zero_counts = win_size - non_zero_counts
            zero_frac = float(zero_counts) / win_size
            #print "zero_frac", zero_fra   
            # skip if image is mostly empty
            if zero_frac >= zero_frac_thresh:
                if verbose:
                    print "Zero frac too high at:", zero_frac
                continue 
            # else save                  
            else:
                #outpath = os.path.join(outdir, out_name + \
                #'|' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) +\
                #'_' + str(pad) + ext)
                outpath = os.path.join(outdir, out_name + \
                '|' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) +\
                '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h) + ext)

                #outpath = os.path.join(outdir, 'slice_' + out_name + \
                #'_' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) +\
                #'_' + str(pad) + '.jpg')
                if verbose:
                    print "outpath:", outpath
                cv2.imwrite(outpath, window_c)
                n_ims_nonull += 1

    print "Num slices:", n_ims, "Num non-null slices:", n_ims_nonull, \
            "sliceHeight", sliceHeight, "sliceWidth", sliceWidth
    print "Time to slice", image_path, time.time()-t0, "seconds"
      
    return

      
    return