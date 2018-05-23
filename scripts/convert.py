# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:55:43 2015

This script is to convert the txt annotation files to appropriate format needed by YOLO 

@author: Guanghan Ning
Email: gnxr9@mail.missouri.edu
"""

import os
from os import walk, getcwd
from PIL import Image

#import pandas as pd
#import xml.etree.ElementTree as ET
#from lxml import objectify, etree


##############
# XML nastiness
#xml = objectify.parse(open(xmlpath))
#
#tree = ET.parse(xmlpath)
#root = tree.getroot()
#children = root.getchildren()
#
#for child in root:
#     print(child.tag, child.attrib)
#
#im_path = 
#
#etree = ET.fromstring(xml) #create an ElementTree object 
##############


# use labelImg to label
#https://github.com/tzutalin/labelImg
#python labelImg.py

def parse_xml(xmlpath):
    #conda install -c asmeurer xmltodict=0.8.3
    import xmltodict
    with open(xmlpath) as fd:
        doc0 = xmltodict.parse(fd.read())
    
    # parse
    doc = doc0['annotation']
    folder = doc['folder']
    filename = doc['filename']
    image_path = doc['path']
    w, h = int(doc['size']['width']), int(doc['size']['height'])
    # get boxes
    box_list = []
    cat_list = []
    # for multimple objects
    if type(doc['object']) == list:
        for ob in doc['object']:
            print "ob", ob
            category = ob['name']
            cat_list.append(category)
            xmin = float(ob['bndbox']['xmin'])
            ymin = float(ob['bndbox']['ymin'])
            xmax = float(ob['bndbox']['xmax'])
            ymax = float(ob['bndbox']['ymax'])
            box_list.append([xmin, xmax, ymin, ymax])
    else:
        ob = doc['object']
        print "ob", ob
        category = ob['name']
        cat_list.append(category)
        xmin = float(ob['bndbox']['xmin'])
        ymin = float(ob['bndbox']['ymin'])
        xmax = float(ob['bndbox']['xmax'])
        ymax = float(ob['bndbox']['ymax'])
        box_list.append([xmin, xmax, ymin, ymax])
        
    print "boxes:", box_list
    print "categories:", cat_list
    return folder, filename, image_path, box_list, cat_list
    
# test
#xmlpath = mypath + 'slice_WV03_03102015_R1C2_Masked_small_2560_1792.xml'
#xmlpath = mypath  + 'slice_WV03_03102015_R1C2_Masked_small_0_0.xml'
#im_path0, box_list0, cat_list0 = parse_xml(xmlpath)  



def convert(size, box):
    '''Input = image size: (w,h), box: [x0, x1, y0, y1]'''
    dw = 1./size[0]
    dh = 1./size[1]
    xmid = (box[0] + box[1])/2.0
    ymid = (box[2] + box[3])/2.0
    w0 = box[1] - box[0]
    h0 = box[3] - box[2]
    x = xmid*dw
    y = ymid*dh
    w = w0*dw
    h = h0*dh
    return (x,y,w,h)
    


def convert_reverse(size, box):
    '''Back out pixel coords from yolo format
    input = image_size (w,h), 
        box = [x,y,w,h]'''
    x,y,w,h = box
    dw = 1./size[0]
    dh = 1./size[1]
    
    w0 = w/dw
    h0 = h/dh
    xmid = x/dw
    ymid = y/dh
    
    x0, x1 = xmid - w0/2., xmid + w0/2.
    y0, y1 = ymid - h0/2., ymid + h0/2.

    return [x0, x1, y0, y1]
    


    
"""-------------------------------------------------------------------""" 

def main(boxroot, mypath, outpath, outname, classes_dic, im_locs_for_list, train_dir):
    
    
    #cls = outname#'boat'
    wd = outname# = getcwd()
    #list_file = open(boxroot + '%s/%s_list.txt'%(wd, cls), 'wb')
    list_file = open(boxroot + '%s_list.txt'%(wd), 'wb')

    
    """ Get input text file list """
    

    txt_name_list = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        txt_name_list.extend(filenames)
        break
    print(txt_name_list)
    
    """ Process """
    for txt_name in txt_name_list:
        # txt_file =  open("Labels/stop_sign/001.txt", "r")
        txt_root = txt_name.split('.')[0]
        
        """ Open input text files """
        txt_path = mypath + txt_name
        print("Input:" + txt_path)
        #txt_file = open(txt_path, "r")
        #lines = txt_file.read().split('\r\n')   #for ubuntu, use "\r\n" instead of "\n"
        folder, filename, img_path, lines, cat_list = parse_xml(txt_path)
        
        """ Open output text files """
        txt_outpath = outpath + txt_root + '.txt'
        print("Output:" + txt_outpath)
        txt_outfile = open(txt_outpath, "w")
        
        
        """ Convert the data to YOLO format """
        ct = 0
        for line, cat in zip(lines, cat_list):
            #print('lenth of line is: ')
            #print(len(line))
            #print('\n')
            if(len(line) >= 2):
                ct = ct + 1
                print "box:", line
                #print(line + "\n")
    #            elems = line.split(' ')
    #            print(elems)
    #            xmin = elems[0]
    #            xmax = elems[2]
    #            ymin = elems[1]
    #            ymax = elems[3]
                xmin,xmax,ymin,ymax = line
                #
                #img_path = str('%s/images/%s/%s.JPEG'%(wd, cls, os.path.splitext(txt_name)[0]))
                ##t = magic.from_file(img_path)
                ##wh= re.search('(\d+) x (\d+)', t).groups()
                
                im_path = train_dir + '/' + filename + '.jpg'
                #im_path = boxroot + '/' + folder + '/' + filename + '.jpg'
                #im_path = img_path
                im=Image.open(im_path)
                
                w= int(im.size[0])
                h= int(im.size[1])
                #w = int(xmax) - int(xmin)
                #h = int(ymax) - int(ymin)
                # print(xmin)
                print(w, h)
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert((w,h), b)
                #print "bb:", bb
                cls_id = classes_dic[cat]
                outstring = str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
                print "outstring:", outstring
                # print "txt_outfile:", txt_outfile
                txt_outfile.write(outstring)
        txt_outfile.close()
        
        """ Save those images with bb into list"""
        if(ct != 0):
            #list_file.write('%s/images/%s/%s.JPEG\n'%(wd, cls, os.path.splitext(txt_name)[0]))
            #list_file.write('%s/images/%s.jpg\n'%(wd, os.path.splitext(txt_name)[0]))
            list_file.write('%s/%s.jpg\n'%(im_locs_for_list, os.path.splitext(txt_name)[0]))

    list_file.close()       

if __name__ == "__main__":
         
    """ Configure Paths""" 
    print ""
        
