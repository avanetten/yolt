#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:21:16 2017

@author: avanetten
"""

import numpy as np
import math
import cv2
import os


###############################################################################    
def rotatePoint(centerPoint, point, angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise
    #http://stackoverflow.com/questions/20023209/python-function-for-rotating-2d-objects
    """
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = (temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle), 
                  temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point
#rotatePoint( (1,1), (2,2), 45)       

###############################################################################    
def rescale_angle(angle_rad):
    '''transform theta to angle between -45 and 45 degrees
    expect input angle to be between 0 and pi radians'''
    angle_deg = round(180. * angle_rad / np.pi, 2)
 
    if angle_deg >= 0. and angle_deg <= 45.:
        angle_out = angle_deg
    elif angle_deg > 45. and angle_deg < 90.:
        angle_out = angle_deg - 90.
    elif angle_deg >= 90. and angle_deg < 135:
        angle_out = angle_deg - 90.
    elif angle_deg >= 135 and angle_deg < 180:
        angle_out = angle_deg - 180.
    else:
        print "Unexpected angle in rescale_angle() [should be from 0-pi radians]"
        return
        
    return angle_out

###############################################################################
def rotate_box(xmin, xmax, ymin, ymax, canny_edges, verbose=False):
    '''Rotate box'''
    
    coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    centerPoint = (np.mean([xmin, xmax]), np.mean([ymin, ymax]))
    
    # get canny edges in desired window
    win_edges = canny_edges[ymin:ymax, xmin:xmax]
    
    # create hough lines
    hough_lines = cv2.HoughLines(win_edges,1,np.pi/180,20)
    if hough_lines is not None:   
        #print "hough_lines:", hough_lines
        # get primary angle
        line = hough_lines[0]
        if verbose:
            print " hough_lines[0]",  line
        if len(line) > 1:
            rho, theta = line[0].flatten()
        else:
            rho, theta = hough_lines[0].flatten()
    else:
        theta = 0.
    # rescale to between -45 and +45 degrees
    angle_deg = rescale_angle(theta)
    if verbose:
        print "angle_deg_rot:", angle_deg
    # rotated coords
    coords_rot = np.asarray([rotatePoint(centerPoint, c, angle_deg) for c in
                             coords], dtype=np.int32)

    return coords_rot


###############################################################################
###############################################################################
# test
if __name__ == "__main__":
    
    '''
    Test rotation
    Plot images show a green box in the center, a red line to show the 
    direction of the first Hough line, and the blue box is just the green
    box rotated to align with the red line
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    '''
    
    indir = '/Users/avanetten/Documents/cosmiq/yolt2/test_data/'
    os.chdir('/Users/avanetten/Documents/cosmiq/yolt2/')
    thickness=2
    
    for i in range(4):
        im_root = '__' + str(i) + '.png'#'rio_airstrip.png'
        im_path = indir + im_root
        
        img = cv2.imread(im_path, 1)
        #gray = cv2.imread(im_path, 0)#cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi/180,50)
        
        # plot lines
        #for l in lines:
        for l in [lines[0]]:
            #for rho,theta in l.flatten():
            for rho,theta in l:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                
                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
                angle_print = int(180. * theta / np.pi)
                print "angle (deg):", angle_print
                
        cv2.imwrite('houghlines' + im_root.split('.')[0] + '_' + str(angle_print) + '.jpg',img)
    
        # create a box at the center
        h,w = gray.shape
        xmid, ymid = w/2, h/2
        l = int(np.min([w,h]) / 2.5)
        xmin, ymin, xmax, ymax = xmid-l, ymid-l, xmid+l, ymid+l
        # plot box
        color0 = (0,255,0)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (color0), thickness)   
        # rotate box
        angle_rescale = rescale_angle(theta)
        print "angle_rescale (deg):", angle_rescale
        coords_rot = rotate_box(xmin, xmax, ymin, ymax, edges)
        # plot
        coords1 = coords_rot.reshape((-1,1,2))
        color1 = (255,0,0)
        cv2.polylines(img, [coords1], True, color1, thickness=thickness)
        
        cv2.imshow('t', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
        
