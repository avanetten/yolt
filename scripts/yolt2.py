 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:11:56 2016

@author: avanetten

# run nvidia-docker nteractive shell 
nvidia-docker run -it -v /raid:/raid —name yolt_name darknet 


##########################
# See notes.txt, yolt_data_prep.py for more info

# for training, yolt replaces "images" with "labels" in path to look for 
#   labels, so paths must be the same except for that 
# training images in yolt2/itraining_datasets/"name"/training_data/images
# training labels must be in in yolt2/itraining_datasets/"name"/training_data/labels

# see data.c.fill_truth_region for assumed data structure

# make sure GPU = 1 in Makefile, run compile_darknet()
# every time C files are changed, need to run compile_darknet()

##########################
# Labeling and bounding box settings
# put all images in yolt2/images/boat
# put all labels in yolt2/labels/boat
# put list of images in yolt2/data/boat_list2_dev_box.txt
##########################
##########################f

"""

import os
import sys
import time
import datetime
import pickle
import cv2
import csv
import pandas as pd
import numpy as np
import argparse
import shutil
from osgeo import ogr
from subprocess import Popen, PIPE, STDOUT
sys.stdout.flush()
#import math
#import random
#from collections import OrderedDict
#import matplotlib.pyplot as plt



###############################################################################
def init_args():
    
    '''Save all variables as in args. Deriive a bunch of values and save in 
    args as well'''
    
    ###########################################################################
    ### Construct argument parser
    parser = argparse.ArgumentParser()
    
    # general settings
    parser.add_argument('--mode', type=str, default='test',
                        help="[compile, test, train, valid]")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU number, set < 0 to turn off GPU support")
    parser.add_argument('--nbands', type=int, default=3,
                        help="Number of input bands (e.g.: for RGB use 3)")
    parser.add_argument('--outname', type=str, default='tmp',
                        help="unique name of output")
    parser.add_argument('--cfg_file', type=str, default='yolo.cfg',
                        help="Configuration file for netowrk, in cfg directory")
    parser.add_argument('--object_labels_str', type=str, default='boat,car',
                        help="Ordered list of objects, will be split into array " \
                              + "by commas (e.g.: 'boat,car' => ['boat','car'])")
    parser.add_argument('--single_gpu_machine', type=int, default=0,
                        help="Switch to use a machine with just one gpu")
    parser.add_argument('--weight_file', type=str, default='yolo.weights',
                        help="Input weight file")
    parser.add_argument('--keep_valid_slices', type=str, default='FALSE',
                        help="Switch to retain sliced valid files")
    

    # training settings
    parser.add_argument('--train_images_list_file', type=str, default='',
                        help="file holding training image names, should be in " \
                            "yolt_dir/data/")
    #parser.add_argument('--train_input_weights', type=str, default='extraction.conv',
    #                    help="Weights to start training with")
    parser.add_argument('--max_batches', type=int, default=60000,
                        help="Max number of training batches")    
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of images per batch")
    parser.add_argument('--subdivisions', type=int, default=4,
                        help="Subdivisions per batch")
        

    # valid settings
    parser.add_argument('--valid_weight_dir', type=str, default='',
                        help="Directory holding trained weights")
    #parser.add_argument('--valid_weight_file', type=str, default='',
    #                    help="Weight file, assumed to exist in valid_weight_dir/")
    parser.add_argument('--valid_testims_dir', type=str, default='',
                        help="Location of test images")
    
    parser.add_argument('--plot_thresh_str', type=str, default='0.3',
                        help="Proposed thresholds to try for valid, will be split"\
                           +" into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])")
    parser.add_argument('--slice_sizes_str', type=str, default='416',
                        help="Proposed pixel slice sizes for valid, will be split"\
                           +" into array by commas (e.g.: '0.2,0.3' => [0.2,0.3])")
    parser.add_argument('--edge_buffer_valid', type=int, default=-1000,
                        help="Buffer around slices to ignore boxes (helps with"\
                            +" truncated boxes and stitching) set <0 to turn off"\
                            +" if not slicing test ims")
    parser.add_argument('--max_edge_aspect_ratio', type=float, default=3,
                        help="Max aspect ratio of any item within the above "\
                            +" buffer")
    parser.add_argument('--slice_overlap', type=float, default=0.35,
                        help="Overlap fraction for sliding window in valid")
    parser.add_argument('--nms_overlap', type=float, default=0.5,
                        help="Overlap threshold for non-max-suppresion in python"\
                            +" (set to 0 to turn off)")
    parser.add_argument('--extra_pkl', type=str, default='',
                        help="External pkl to load on plots")
    parser.add_argument('--c_nms_thresh', type=float, default=0.0,
                        help="Defaults to 0.5 in yolt2.c, set to 0 to turn off "\
                            +" nms in C")
    parser.add_argument('--valid_box_rescale_frac', type=float, default=1.0,
                        help="Defaults to 1, rescale output boxes if training"\
                            + " boxes are the wrong size")    
    parser.add_argument('--valid_make_pngs', type=str, default='FALSE',
                        help="Switch to save validation pngs")
    parser.add_argument('--valid_make_legend_and_title', type=str, default='TRUE',
                        help="Switch to make legend and title")    
    parser.add_argument('--valid_im_compression_level', type=int, default=6,
                        help="Compression level for output images."\
                            + " 1-9 (9 max compression")    
    parser.add_argument('--make_valid_pkl', type=int, default=1,
                        help="Switch to make valid pickles")
    
    # test settings
    parser.add_argument('--test_im', type=str, default='person.jpg',
                        help="test image, in data_dir")
    parser.add_argument('--test_thresh', type=float, default=0.2,
                        help="prob thresh for plotting outputs")
    parser.add_argument('--test_labels', type=str, default='coco.names',
                        help="test labels, in data_dir")
    #parser.add_argument('--show_test_labels', type=int, default=0,
    #                    help="switch to show test labels in inference, 0 == off")
    
    
    # Defaults that rarely should need changed
    parser.add_argument('--yolt_dir', type=str, default='/raid/local/src/yolt2/',
                        help="path to package")
    parser.add_argument('--use_opencv', type=str, default='1',
                        help="1 == use_opencv")
    parser.add_argument('--boxes_per_grid', type=int, default=5,
                        help="Bounding boxes per grid cell")
    parser.add_argument('--multi_band_delim', type=str, default='#',
                        help="Delimiter for multiband data")
    parser.add_argument('--zero_frac_thresh', type=float, default=0.5,
                        help="If less than this value of an image chip is blank,"\
                            + " skip it")
    parser.add_argument('--show_valid_plots', type=int, default=0,
                        help="Switch to show plots in real time in validation")
    parser.add_argument('--plot_names', type=int, default=0,
                        help="Switch to show plots names in validation")
    parser.add_argument('--rotate_boxes', type=int, default=0,
                        help="Attempt to rotate output boxes using hough lines")
    parser.add_argument('--plot_line_thickness', type=int, default=2,
                        help="Thickness for valid output bounding box lines")
    parser.add_argument('--str_delim', type=str, default=',',
                        help="Delimiter for string lists")
    
    # placeholder, values are dynsmic will be set later
    parser.add_argument('--valid_files_txt', type=str, default='',
                        help="File containing location of sliced images")
    parser.add_argument('--valid_results_files', type=str, default='',
                        help="List of validation results  files")
    parser.add_argument('--valid_split_dir_list', type=str, default='',
                        help="List of directories of sliced validation images")
    
    args = parser.parse_args()
       
    
    ###########################################################################
    # CONSTRUCT INFERRED VALUES
    ###########################################################################
    
    ##########################
    # GLOBAL VALUES
    # set directory structure
    # append '/' to end of yolt_dir
    if not args.yolt_dir.endswith('/'): args.yolt_dir += '/'
    args.results_topdir = args.yolt_dir + 'results/'
    args.test_images_dir = args.yolt_dir + 'test_images/'
    args.data_dir = args.yolt_dir + 'data/'
    args.weight_dir = args.yolt_dir + 'input_weights/'
    args.cfg_dir = args.yolt_dir + 'cfg/'
    args.label_image_dir = args.data_dir + 'category_label_images/'
    args.this_file = args.yolt_dir + 'scripts/yolt2.py'
    args.plot_file = args.yolt_dir + 'scripts/yolt_plot_loss.py'

    args.extension_list = ['.png', '.tif', '.TIF', '.TIFF', '.tiff', '.JPG', '.jpg', '.JPEG', '.jpeg']
        
    # infer lists from args
    args.object_labels = args.object_labels_str.split(args.str_delim)
    args.plot_thresh = np.array(args.plot_thresh_str.split(args.str_delim)).astype(float)
    args.slice_sizes = np.array(args.slice_sizes_str.split(args.str_delim)).astype(int)
            
    # set training files
    args.train_images_list_file_tot = os.path.join(args.data_dir, args.train_images_list_file)
        
    # set validation files
    # first prepend paths to directories
    args.valid_weight_dir_tot = os.path.join(args.results_topdir, args.valid_weight_dir)
    #args.valid_weight_file_tot = args.valid_weight_dir + args.valid_weight_file
    args.valid_testims_dir_tot = os.path.join(args.test_images_dir, args.valid_testims_dir)
    # set test list 
    try:
        if args.nbands == 3:
            args.valid_testims_list = [f for f in os.listdir(args.valid_testims_dir_tot) \
                                       if f.endswith(tuple(args.extension_list))]
        else:
            args.valid_testims_list = [f for f in os.listdir(args.valid_testims_dir_tot) \
                                       if f.endswith('#1.png')]
    except:
        args.valid_testims_list = []
    
    
    # set side length from cfg root, classnum, and final output
    # default to length of 13
    try:
        args.side = int(args.cfg_file.split('.')[0].split('x')[-1])  # Grid size (e.g.: side=20 gives a 20x20 grid)
    except:
        args.side = 13
    args.classnum = len(args.object_labels)
    args.final_output = 1 * 1 * args.boxes_per_grid * (args.classnum + 4 + 1)
    
    # set cuda values
    if args.gpu >= 0:
        args.use_GPU, args.use_CUDNN = 1, 1
    else:
        args.use_GPU, args.use_CUDNN = 0, 0
    
    # test settings, assume test images are in test_images
    args.test_im_tot = os.path.join(args.test_images_dir, args.test_im)
    #args.testweight_file = args.weight_dir + args.testweight_file
    # populate test_labels
    test_labels_list = []
    if args.mode == 'test':
        # populate labels
        with open(os.path.join(args.data_dir, args.test_labels), 'rb') as fin:
            for l in fin.readlines():
                # spaces in names screws up the argc in yolt2.c
                test_labels_list.append(l[:-1].replace(' ', '_'))
        # overwrite object_labels, and object_labels_str
        args.object_labels = test_labels_list
        args.object_labels_str = ','.join([str(ltmp) for ltmp in \
                                           test_labels_list])
    
    
    ##########################################
    # Get datetime and set outlog file
    args.now = datetime.datetime.now()
    args.date_string = args.now.strftime('%Y_%m_%d_%H-%M-%S')
    #print "Date string:", date_string
    args.res_name = args.mode + '_' + args.outname + '_cfg=' \
                        + args.cfg_file.split('.')[0] + '_' + args.date_string
    args.results_dir = os.path.join(args.results_topdir, args.res_name) #+ '/'
    args.log_dir = os.path.join(args.results_dir, 'logs')#/'
    args.log_file = os.path.join(args.log_dir, args.res_name + '.log')
    args.loss_file = os.path.join(args.log_dir, 'loss.txt')

    ## make dirs (do this in main() below)
    #os.mkdir(args.results_dir)
    #os.mkdir(args.log_dir)
    
    # set cfg_file, assume raw cfgs are in cfg directory, and the cfg file
    #   will be copied to log_dir.  Use the cfg in logs as input to yolt2.c
    #   with valid, cfg will be in results_dir/logs/
    # if using valid, assume cfg file is in valid_weight_dir, else, assume
    #   it's in yolt_dir/cfg/
    # also set weight file
    args.cfg_file_tot = os.path.join(args.log_dir, args.cfg_file)
    if args.mode == 'valid':
        # assume weights and cfg are in the training dir
        args.cfg_file_in = os.path.join(args.valid_weight_dir_tot, 'logs/', args.cfg_file)
        args.weight_file_tot = os.path.join(args.valid_weight_dir_tot, args.weight_file)
    else:
        # assume weights are in weight_dir, and cfg in cfg_dir
        args.cfg_file_in = os.path.join(args.cfg_dir, args.cfg_file)
        args.weight_file_tot = os.path.join(args.weight_dir, args.weight_file)

    ## set batch size based on network size?
    #if args.side >= 30:
    #    args.batch_size = 16                 # batch size (64 for 14x14, 32 for 28x28)
    #    args.subdivisions = 8               # subdivisions per batch (8 for 14x14 [yields 8 images per batch], 32 for 28x28)
    #elif args.side >= 16:
    #    args.batch_size = 64                 # batch size (64 for 14x14, 32 for 28x28)
    #    args.subdivisions = 16               # subdivisions per batch (8 for 14x14 [yields 8 images per batch], 32 for 28x28)
    #else:
    #    args.batch_size = 64
    #    args.subdivisions = 8
    
    ##########################
    # Plotting params
    args.figsize = (12,12)
    args.dpi = 300
    
    return args

###############################################################################
def file_len(fname):
    '''Return length of file'''
    try:
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    except:
        return 0
    
###############################################################################
def run_cmd(cmd):
    '''Write to stdout, etc, THIS SCREWS UP NOHUP.OUT!!!'''
    p = Popen(cmd, stdout = PIPE, stderr = STDOUT, shell = True)
    while True:
        line = p.stdout.readline()
        if not line: break
        print line.replace('\n', '') 
    return

###############################################################################
def yolt_command(args):
    
    '''
    Define YOLT commands
    yolt2.c expects the following inputs:
    // arg 0 = GPU number
    // arg 1 'yolt'
    // arg 2 = mode
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *test_filename = (argc > 5) ? argv[5]: 0;
    float plot_thresh = (argc > 6) ? atof(argv[6]): 0.2;
    float nms_thresh = (argc > 7) ? atof(argv[7]): 0;
    char *train_images = (argc > 8) ? argv[8]: 0;
    char *results_dir = (argc > 9) ? argv[9]: 0;
    //char *valid_image = (argc >10) ? argv[10]: 0;
    char *valid_list_loc = (argc > 10) ? argv[10]: 0;
    char *names_str = (argc > 11) ? argv[11]: 0;
    int len_names = (argc > 12) ? atoi(argv[12]): 0;
    int nbands = (argc > 13) ? atoi(argv[13]): 0;
    char *loss_file = (argc > 14) ? argv[14]: 0;
    '''

    ##########################
    # set gpu command
    if args.single_gpu_machine == 1:#use_aware:
        gpu_cmd = ''
    else:
        gpu_cmd = '-i ' + str(args.gpu) 
        #gpu_cmd = '-i ' + str(3-args.gpu)    # originally, numbers were reversed

    ##########################
    # SET VARIABLES ACCORDING TO MODE (SET UNNECCESSARY VALUES TO 0 OR NULL)      
    # set train prams (and prefix, and suffix)
    if args.mode == 'train':
        train_ims = args.train_images_list_file_tot
        prefix = 'nohup'
        suffix = ' >> ' + args.log_file + ' & tail -f ' + args.log_file
    else:
        train_ims = 'null'
        prefix = ''
        suffix =  ' 2>&1 | tee -a ' + args.log_file
        
    # set test params
    if args.mode == 'test':
        test_im = args.test_im_tot
        test_thresh = args.test_thresh
    else:
        test_im = 'null'
        test_thresh = 0
    
    # set valid params
    if args.mode == 'valid':
        #valid_image = args.valid_image_tmp
        valid_list_loc = args.valid_files_txt
    else:
        #valid_image = 'null'
        valid_list_loc = 'null'
            
    
    ##########################

    c_arg_list = [
            prefix,
            './darknet',
            gpu_cmd,
            'yolt2',
            args.mode,
            args.cfg_file_tot,
            args.weight_file_tot,
            test_im,
            str(test_thresh),
            str(args.c_nms_thresh),
            train_ims,
            args.results_dir,
            #valid_image,
            valid_list_loc,
            args.object_labels_str,
            str(args.classnum),
            str(args.nbands),
            args.loss_file,
            suffix
            ]
            
    cmd = ' '.join(c_arg_list)
      
    print "Command:\n", cmd        
      
    return cmd
    

###############################################################################
def make_label_images(root_dir, new_labels=[]):
    '''Create new images of label names'''
        
    # legacy0
    l0 = ["person","bicycle","car","motorcycle","airplane","bus","train",
          "truck","boat","traffic light","fire hydrant","stop sign",
          "parking meter","bench","bird","cat","dog","horse","sheep","cow",
          "elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
          "tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
          "baseball bat","baseball glove","skateboard","surfboard",
          "tennis racket","bottle","wine glass","cup","fork","knife","spoon",
          "bowl","banana","apple","sandwich","orange","broccoli","carrot",
          "hot dog","pizza","donut","cake","chair","couch","potted plant",
          "bed","dining table","toilet","tv","laptop","mouse","remote",
          "keyboard","cell phone","microwave","oven","toaster","sink",
          "refrigerator","book","clock","vase","scissors","teddy bear",
          "hair drier","toothbrush", "aeroplane", "bicycle", "bird", "boat", 
          "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
          "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
          "train", "tvmonitor"]
    
    # legacyl
    l1 = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
          "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
          "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    coco2 = ["person","bicycle","car","motorcycle","airplane","bus","train",
          "truck","boat","traffic_light","fire_hydrant","stop_sign",
          "parking_meter","bench","bird","cat","dog","horse","sheep","cow",
          "elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
          "tie","suitcase","frisbee","skis","snowboard","sports_ball","kite",
          "baseball_bat","baseball_glove","skateboard","surfboard",
          "tennis_racket","bottle","wine_glass","cup","fork","knife","spoon",
          "bowl","banana","apple","sandwich","orange","broccoli","carrot",
          "hot_dog","pizza","donut","cake","chair","couch","potted plant",
          "bed","dining_table","toilet","tv","laptop","mouse","remote",
          "keyboard","cell_phone","microwave","oven","toaster","sink",
          "refrigerator","book","clock","vase","scissors","teddy_bear",
          "hair_drier","toothbrush"]
    
    # new
    l2 = ["boat", "dock", "boat_harbor", "airport", "airport_single", 
          "airport_multi"]
    
    l = l0 + l1 + coco2 + l2 + new_labels
    
    #for word in l:
    #    os.system("convert -fill black -background white -bordercolor white "\
    #               +"-border 4 -font futura-normal -pointsize 18 label:\"%s\" \"%s.jpg\""%(word, word))
    
    # change to label directory
    cwd = os.getcwd()
    os.chdir(root_dir)
    for word in l:
        #os.system("convert -fill black -background white -bordercolor white -border 4 -font Helvetica -pointsize 18 label:\"%s\" \"%s.jpg\""%(word, word))
        run_cmd("convert -fill black -background white -bordercolor white "\
                + "-border 4 -font Helvetica -pointsize 18 label:\"%s\" \"%s.jpg\""%(word, word))
        run_cmd("convert -fill black -background white -bordercolor white "\
                + "-border 4 -font Helvetica -pointsize 18 label:\"%s\" \"%s.png\""%(word, word))

    # change back to cwd
    os.chdir(cwd)  
    return

###############################################################################
def recompile_darknet(yolt_dir):
    '''compile darknet'''
    os.chdir(yolt_dir)
    cmd_compile0 = 'make clean'
    cmd_compile1 = 'make'
    
    print cmd_compile0
    run_cmd(cmd_compile0)
    
    print cmd_compile1
    run_cmd(cmd_compile1)    
    
###############################################################################
def replace_yolt_vals(args):

    '''edit cfg file in darknet to allow for custom models
    editing of network layers must be done in vi, this function just changes 
    parameters such as window size, number of trianing steps, etc'''
        
    #################
    # Makefile
    if args.mode == 'compile':
        yoltm = os.path.join(args.yolt_dir, 'Makefile')
        yoltm_tmp = yoltm + 'tmp'
        f1 = open(yoltm, 'r')
        f2 = open(yoltm_tmp, 'w')   
        for line in f1:
            if line.strip().startswith('GPU='):
                line_out = 'GPU=' + str(args.use_GPU) + '\n'
            elif line.strip().startswith('OPENCV='):
                line_out = 'OPENCV=' + str(args.use_opencv) + '\n'
            elif line.strip().startswith('CUDNN='):
                line_out = 'CUDNN=' + str(args.use_CUDNN) + '\n'            
            else:
                line_out = line
            f2.write(line_out)
        f1.close()
        f2.close()    
        # copy old yoltm
        run_cmd('cp ' + yoltm + ' ' + yoltm + '_v0')
        # write new file over old
        run_cmd('mv ' + yoltm_tmp + ' ' + yoltm)

    #################
    # cfg file
    elif args.mode == 'train':
        yoltcfg = args.cfg_file_tot
        yoltcfg_tmp = yoltcfg + 'tmp'
        f1 = open(yoltcfg, 'r')
        f2 = open(yoltcfg_tmp, 'w')    
        # read in reverse because we want to edit the last output length
        s = f1.readlines()
        s.reverse()
        sout = []
        
        fixed_output = False
        for line in s:
            #if line.strip().startswith('side='):
            #    line_out='side=' + str(side) + '\n'
            if line.strip().startswith('channels='):
                line_out = 'channels=' + str(args.nbands) + '\n'
            elif line.strip().startswith('classes='):
                line_out = 'classes=' + str(args.classnum) + '\n'
            elif line.strip().startswith('max_batches'):
                line_out = 'max_batches=' + str(args.max_batches) + '\n'
            elif line.strip().startswith('batch='):
                line_out = 'batch=' + str(args.batch_size) + '\n'  
            elif line.strip().startswith('subdivisions='):
                line_out = 'subdivisions=' + str(args.subdivisions) + '\n'  
            elif line.strip().startswith('num='):
                line_out = 'num=' + str(args.boxes_per_grid) + '\n'         
            # change final output, and set fixed to true
            #elif (line.strip().startswith('output=')) and (not fixed_output):
            #    line_out = 'output=' + str(final_output) + '\n'
            #    fixed_output=True
            elif (line.strip().startswith('filters=')) and (not fixed_output):
                line_out = 'filters=' + str(args.final_output) + '\n'
                fixed_output=True                
            else:
                line_out = line
            sout.append(line_out)
            
        sout.reverse()
        for line in sout:
            f2.write(line)
           
        f1.close()
        f2.close()
        
        # copy old yoltcfg?
        run_cmd('cp ' + yoltcfg + ' ' + yoltcfg[:-4] + 'orig.cfg')
        # write new file over old
        run_cmd('mv ' + yoltcfg_tmp + ' ' + yoltcfg)    
    #################
   
    else:
        return

    
###############################################################################
def split_valid_im(im_root_with_ext, args):
    
    '''split files for valid step
    Assume input string has no path, but does have extension (e.g:, 'pic.png')
    
    1. get image path (args.valid_image_tmp) from image root name 
            (args.valid_image_tmp)
    2. slice test image and move to results dir

    '''

    ##########################
    # import slice_im
    sys.path.append(os.path.join(args.yolt_dir, 'scripts'))
    import slice_im
    ##########################

    
    # get image root, make sure there is no extension
    im_root = im_root_with_ext.split('.')[0]
    im_path = os.path.join(args.valid_testims_dir_tot, im_root_with_ext)
    
    # slice validation plot into manageable chunks
    
    # slice (if needed)
    if args.slice_sizes[0] > 0:
    #if len(args.slice_sizes) > 0:
        # create valid_files_txt 
        # set valid_dir as in results_dir
        # if multi_band_delim = '#' in valid_image, remove suffix of '#1', as
        #   the splitting function in image.c screws this up
        if args.multi_band_delim in im_root:
            valid_split_dir = os.path.join(args.results_dir, im_root[:-2] + '_split' + '/')
        else:
            valid_split_dir = os.path.join(args.results_dir,  im_root + '_split' + '/')
            
        valid_dir_str = '"Valid_split_dir: ' +  valid_split_dir + '\n"'
        print valid_dir_str[1:-2]
        os.system('echo ' + valid_dir_str + ' >> ' + args.log_file)
        #print "valid_split_dir:", valid_split_dir
        
        # clean out dir, and make anew
        if os.path.exists(valid_split_dir):
            shutil.rmtree(valid_split_dir, ignore_errors=True)
        os.mkdir(valid_split_dir)

        for s in args.slice_sizes:
            if args.multi_band_delim in im_root:
                # UNTESTED!
                for im_root_tmp in [im_root_with_ext, im_root[:-1]+'2.png', \
                                    im_root[:-1]+'3.png']:
                    #path_tmp = args.valid_testims_dir_tot + im_root_tmp
                    slice_im.slice_im(im_path, im_root_tmp, 
                              valid_split_dir, s, s, 
                              zero_frac_thresh=args.zero_frac_thresh, 
                              overlap=args.slice_overlap)
                valid_files = [valid_split_dir + f for f in \
                           os.listdir(valid_split_dir) if f.endswith('1.png')]
            else:
                slice_im.slice_im(im_path, im_root, 
                              valid_split_dir, s, s, 
                              zero_frac_thresh=args.zero_frac_thresh, 
                              overlap=args.slice_overlap)
                valid_files = [os.path.join(valid_split_dir, f) for \
                                   f in os.listdir(valid_split_dir)]
        n_files_str = '"Num files: ' + str(len(valid_files)) + '\n"'
        print n_files_str[1:-2]
        os.system('echo ' + n_files_str + ' >> ' + args.log_file)
        
    else:
        valid_files = [im_path]
        valid_split_dir = os.path.join(args.results_dir, 'nonsense')

    return valid_files, valid_split_dir


###############################################################################
def set_valid_files(valid_files_list, args):
    
    '''Set files for valid step
    Assume input string has no path, but does have extension (e.g:, 'pic.png')
    
    3. create .txt file of test image locations (args.valid_files_txt)
    4. create list of valid results files 
    '''
    
    print "Total len valid files:", len(valid_files_list)
    valid_files_txt = os.path.join(args.results_dir, 'valid_input_files.txt')
    print "valid_files_txt:", valid_files_txt
    # write list of files to valid_files_txt
    with open (valid_files_txt, "wb") as fp:
       for line in valid_files_list:
           if not line.endswith('.DS_Store'):
               fp.write(line + "\n")
               
    # yolt2.c puts val files in darknet dir by default, but this
    #   has been updated        
    valid_results_files = [os.path.join(args.results_dir, l + '.txt') \
                                           for l in args.object_labels]
    print "valid_results_files:", valid_results_files
    
    return valid_files_txt, valid_results_files

###############################################################################
def run_valid(args):
    '''Evaluate multiple large images'''
    
    # run for each image
    #t00 = time.time()    
    
    # split validation images, store locations 
    valid_split_str = '"Splitting validation files...\n"'
    print valid_split_str[1:-2]
    os.system('echo ' + valid_split_str + ' >> ' + args.log_file)

    valid_files_locs_list = []
    valid_split_dir_list = []
    for i,valid_base_tmp in enumerate(args.valid_testims_list):
        iter_string = '"\n' + str(i+1) + ' / ' + \
            str(len(args.valid_testims_list)) + '\n"'
        print iter_string[1:-2]
        os.system('echo ' + iter_string + ' >> ' + args.log_file)
        #print "\n", i+1, "/", len(args.valid_testims_list)
        
        # dirty hack: ignore file extensions for now
        #valid_base_tmp_noext = valid_base_tmp.split('.')[0]
        #valid_base_string = '"valid_base_tmp_noext:' \
        #                    + str(valid_base_tmp_noext) + '\n"'
        valid_base_string = '"valid_file: ' + str(valid_base_tmp) + '\n"'
        print valid_base_string[1:-2]
        os.system('echo ' + valid_base_string + ' >> ' + args.log_file)
        
        # split data 
        valid_files_list_tmp, valid_split_dir_tmp = split_valid_im(valid_base_tmp, args)
        # add valid_files to list
        valid_files_locs_list.extend(valid_files_list_tmp)
        valid_split_dir_list.append(valid_split_dir_tmp)
        
    # set all validation files
    valid_files_txt, valid_results_files = set_valid_files(valid_files_locs_list, 
                                                           args)
      
    # add to args
    args.valid_files_txt = valid_files_txt
    args.valid_results_files = valid_results_files
    args.valid_split_dir_list = valid_split_dir_list
    #print "args.valid_split_dir_list:", args.valid_split_dir_list
    #print "valid_files_locs_list:", valid_files_locs_list


    # define outcmd
    outcmd = yolt_command(args)

    t0 = time.time()
    # run command
    os.system(outcmd)       #run_cmd(outcmd)
    t1 = time.time()
    n_files = file_len(args.valid_files_txt)
    cmd_time_str = '"\nLength of time to run command: ' +  outcmd \
                    + ' for ' + str(n_files) + ' cutouts: ' \
                    + str(t1 - t0) + ' seconds\n"'
    print cmd_time_str  
    os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)

    # process outputs
    df_tot = post_process_valid_create_df(args)
    # save df_tot
    df_outfile = os.path.join(args.results_dir, '000_predictions_df.csv')
    df_tot.to_csv(df_outfile)
    
    
    
    ###########################################
    
    post_proccess_make_plots(args, df_tot, verbose=True)
    
    # remove or zip valid_split_dirs to save space
    for valid_split_dir_tmp in args.valid_split_dir_list:
        if os.path.exists(valid_split_dir_tmp):
            # compress image chip dirs if desired
            if args.keep_valid_slices.upper() == 'TRUE':
                print "Compressing image chips..."
                shutil.make_archive(valid_split_dir_tmp, 'zip', 
                                    valid_split_dir_tmp)            
            # remove unzipped folder
            print "Removing valid_split_dir_tmp:", valid_split_dir_tmp
            # make sure that valid_split_dir_tmp hasn't somehow been shortened
            #  (don't want to remove "/")
            if len(valid_split_dir_tmp) < len(args.results_dir):
                print "valid_split_dir_tmp too short!!!!:", valid_split_dir_tmp
                return
            else:
                shutil.rmtree(valid_split_dir_tmp, ignore_errors=True)
                
    ## zip image files
    #print "Zipping image files..."
    #for f in os.listdir(args.results_dir):
    #    print "file:", f
    #    if f.endswith(args.extension_list):
    #        ftot = os.path.join(args.results_dir, f)
    #        os.system('gzip ' + ftot)
            
    return

###############################################################################
def get_global_coords(args, row):
    '''Get global coords of bounding box prediction from dataframe row
            #columns:Index([u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin', 
            #                u'Xmax', u'Ymax', u'Category', 
            #                u'Image_Root_Plus_XY', u'Image_Root', u'Slice_XY', 
            #                u'Upper', u'Left', u'Height', u'Width', u'Pad', 
            #                u'Image_Path']
    '''

    xmin0, xmax0 = row['Xmin'], row['Xmax']
    ymin0, ymax0 = row['Ymin'], row['Ymax']
    upper, left = row['Upper'], row['Left']
    sliceHeight, sliceWidth = row['Height'], row['Width']
    vis_w, vis_h = row['Im_Width'], row['Im_Height']
    pad = row['Pad']
    
    ## skip if near edge (set edge_buffer_valid < 0 to skip)
    #if args.edge_buffer_valid > 0:
    #    if ((float(xmin0) < args.edge_buffer_valid) or 
    #        (float(xmax0) > (sliceWidth - args.edge_buffer_valid)) or                      
    #        (float(ymin0) < args.edge_buffer_valid) or 
    #        (float(ymax0) > (sliceHeight - args.edge_buffer_valid)) ):
    #        print "Too close to edge, skipping", row, "..."
    #        return [], []

    # skip if near edge and high aspect ratio (set edge_buffer_valid < 0 to skip)
    if args.edge_buffer_valid > 0:
        if ((float(xmin0) < args.edge_buffer_valid) or 
                (float(xmax0) > (sliceWidth - args.edge_buffer_valid)) or                      
                (float(ymin0) < args.edge_buffer_valid) or 
                (float(ymax0) > (sliceHeight - args.edge_buffer_valid)) ):
            # compute aspect ratio
            dx = xmax0 - xmin0
            dy = ymax0 - ymin0
            if (1.*dx/dy > args.max_edge_aspect_ratio) or (1.*dy/dx > args.max_edge_aspect_ratio):
                print "Too close to edge, and high aspect ratio, skipping", row, "..."
                return [], []
    
        # set min, max x and y for each box, shifted for appropriate
        #   padding                
        xmin = max(0, int(round(float(xmin0)))+left - pad) 
        xmax = min(vis_w, int(round(float(xmax0)))+left - pad)
        ymin = max(0, int(round(float(ymin0)))+upper - pad)
        ymax = min(vis_h, int(round(float(ymax0)))+upper - pad)
    
    else:
        xmin, xmax = xmin0, xmax0
        ymin, ymax = ymin0, ymax0
        
    # rescale output box size if desired, might want to do this
    #    if the training boxes were the wrong size
    if args.valid_box_rescale_frac != 1.0:
        dl = args.valid_box_rescale_frac
        xmid, ymid = np.mean([xmin, xmax]), np.mean([ymin, ymax])
        dx = dl*(xmax - xmin) / 2
        dy = dl*(ymax - ymin) / 2
        x0 = xmid - dx
        x1 = xmid + dx
        y0 = ymid - dy
        y1 = ymid + dy
        xmin, xmax, ymin, ymax = x0, x1, y0, y1

    # rotate boxes, if desird
    if args.rotate_boxes:
        # import post_process scripts
        sys.path.append(os.path.join(args.yolt_dir, 'scripts'))
        import yolt_post_process
    
        # import vis            
        vis = cv2.imread(row['Image_Path'], 1)  # color
        #vis_h,vis_w = vis.shape[:2]
        gray = cv2.cvtColor(vis,cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        #lines = cv2.HoughLines(edges,1,np.pi/180,50)
        coords = yolt_post_process.rotate_box(xmin, xmax, ymin, 
                                              ymax, canny_edges)  

    # set bounds, coords
    bounds = [xmin, xmax, ymin, ymax]
    coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    return bounds, coords        
      

###############################################################################
def post_process_valid_create_df(args):
    '''take output files and create df
    # df.columns:
        # Index([u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin', u'Xmax', u'Ymax', u'Category',
        # u'Image_Root_Plus_XY', u'Image_Root', u'Slice_XY', u'Upper', u'Left',
        # u'Height', u'Width', u'Pad', u'Image_Path'],
        # dtype='object')
        
    '''
 
    # parse out files, create df
    df_tot = []
    
    #str0 = '"args.valid_results_files: ' + str(args.valid_results) + '\n"'
    
    for i,vfile in enumerate(args.valid_results_files):

        valid_base_string = '"valid_file: ' + str(vfile) + '\n"'
        print valid_base_string[1:-2]
        os.system('echo ' + valid_base_string + ' >> ' + args.log_file)
        
        cat = vfile.split('/')[-1].split('.')[0]
        # load into dataframe
        df = pd.read_csv(vfile, sep=' ', names=['Loc_Tmp', 'Prob', 
                                                       'Xmin', 'Ymin', 'Xmax',
                                                       'Ymax'])
        # set category
        df['Category'] = len(df) * [cat]
        # extract image root
        df['Image_Root_Plus_XY'] = [f.split('/')[-1] for f in df['Loc_Tmp']]

        #print "df:", df
        # parse out image root and location
        im_roots, im_locs = [], []
        for j,f in enumerate(df['Image_Root_Plus_XY'].values):
            ext = f.split('.')[-1]
            # get im_root, (if not slicing ignore '|')
            if args.slice_sizes[0] > 0:
                im_root_tmp = f.split('|')[0]
                xy_tmp = f.split('|')[-1]
            else:
                im_root_tmp, xy_tmp = f, '0_0_0_0_0_0_0'

            if im_root_tmp == xy_tmp:
                #xy_tmp = '0_0_0_0_0'
                xy_tmp = '0_0_0_0_0_0_0'
            im_locs.append(xy_tmp)   
            if not '.' in im_root_tmp:
                im_roots.append(im_root_tmp + '.' + ext)
            else:
                im_roots.append(im_root_tmp)

        df['Image_Root'] = im_roots
        df['Slice_XY'] = im_locs
        # get positions
        df['Upper'] = [float(sl.split('_')[0]) for sl in df['Slice_XY'].values]
        df['Left'] = [float(sl.split('_')[1]) for sl in df['Slice_XY'].values]
        df['Height'] = [float(sl.split('_')[2]) for sl in df['Slice_XY'].values]
        df['Width'] = [float(sl.split('_')[3]) for sl in df['Slice_XY'].values]
        df['Pad'] = [float(sl.split('_')[4].split('.')[0]) for sl in df['Slice_XY'].values]
        df['Im_Width'] = [float(sl.split('_')[5].split('.')[0]) for sl in df['Slice_XY'].values]
        df['Im_Height'] = [float(sl.split('_')[6].split('.')[0]) for sl in df['Slice_XY'].values]
        
        # set image path
        df['Image_Path'] = [os.path.join(args.valid_testims_dir_tot, f) for f
                            in df['Image_Root'].values]

        # add in global location of each row
        x0l, x1l, y0l, y1l = [], [], [], []
        bad_idxs = []
        for index, row in df.iterrows():
            bounds, coords = get_global_coords(args, row)
            if len(bounds) == 0 and len(coords) == 0:
                bad_idxs.append(index)
                [xmin, xmax, ymin, ymax] = 0, 0, 0, 0
            else:
                [xmin, xmax, ymin, ymax] = bounds
            x0l.append(xmin)
            x1l.append(xmax)
            y0l.append(ymin)
            y1l.append(ymax)
        df['Xmin_Glob'] = x0l
        df['Xmax_Glob'] = x1l
        df['Ymin_Glob'] = y0l
        df['Ymax_Glob'] = y1l
        
        # remove bad_idxs
        if len(bad_idxs) > 0:
            print "removing bad idxs:", bad_idxs
            df = df.drop(df.index[bad_idxs])        
            
        # append to total df
        if i == 0:
            df_tot = df
        else:
            df_tot = df_tot.append(df, ignore_index=True)
       
    return df_tot
          

###############################################################################
def post_proccess_make_plots(args, df, verbose=False):
    
    # create dictionary of plot_thresh lists
    thresh_poly_dic = {}
    for plot_thresh_tmp in args.plot_thresh:        
        # initilize to empty
        thresh_poly_dic[np.around(plot_thresh_tmp, decimals=2)] = []
    if verbose:
        print "thresh_poly_dic:", thresh_poly_dic
        
    # group 
    group = df.groupby('Image_Path')
    for itmp,g in enumerate(group):
        im_path = g[0]
        im_root_noext = im_path.split('/')[-1].split('.')[0]
        data_all_classes = g[1]
        
            
        print "\n", itmp, "/", len(group), "Analyzing Image:", im_path

        # plot validation outputs    
        #for plot_thresh_tmp in plot_thresh:
        for plot_thresh_tmp in thresh_poly_dic.keys():
            if args.valid_make_pngs.upper() != 'FALSE':
                figname_val = os.path.join(args.results_dir, im_root_noext \
                            + '_valid_thresh=' + str(plot_thresh_tmp) + '.png')
                            #+ '_valid_thresh=' + str(plot_thresh_tmp) + '.jpeg')

            else:
                figname_val = ''
            
            if args.make_valid_pkl == 1:
                pkl_val = os.path.join(args.results_dir, im_root_noext \
                        + '_boxes_thresh=' + str(plot_thresh_tmp) + '.pkl')
            else:
                pkl_val = ''

            ############
            # make plot
            out_list = plot_vals(args, im_path, data_all_classes, pkl_val, 
                                 figname_val, plot_thresh_tmp)                                  
            ############
            
            # convert to wkt format for buildings
            if (len(args.object_labels)==1) and \
                                    (args.object_labels[0]=='building'):
                for i,row in enumerate(out_list):
                    [xmin, ymin, xmax, ymax, coords, filename, textfile, prob, 
                         color0, color1, color2, labeltmp, labeltmp_full] = row
                    im_name0 = filename.split('/')[-1].split('.')[0]
                    if args.multi_band_delim in im_root_noext:
                        im_name1 = im_name0[15:].split('#')[0]
                    else:
                       # im_name1 = im_name0[6:]
                        im_name1 = im_name0[15:]
                    wkt_row = building_polys_to_csv(im_name1, str(i), 
                                                    coords,
                                                    #[xmin,ymin,xmax,ymax],
                                                    conf=prob)                      
                    thresh_poly_dic[plot_thresh_tmp].append(wkt_row)
            
            # save out_list
            if len(out_list) > 0:
                out_list_f = os.path.join(args.results_dir, im_root_noext \
                            + '_plot_vals_outlist.csv')
                with open(out_list_f, "wb") as f:
                    writer = csv.writer(f)
                    writer.writerows(out_list)
                

    # save thresh_poly_dic
    if len(args.object_labels) == 1 and args.object_labels[0] == 'building':
        for plot_thresh_tmp in thresh_poly_dic.keys():
            csv_name = os.path.join(args.results_dir, 'predictions_' \
                                        + str(plot_thresh_tmp) + '.csv')
            print "Saving wkt buildings to file:", csv_name, "..."
            # save to csv
            #print "thresh_poly_dic:", thresh_poly_dic
            #print "thresh_poly_dic[plot_thresh_tmp]:", thresh_poly_dic[plot_thresh_tmp]
            with open(csv_name, 'wb') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for j,line in enumerate(thresh_poly_dic[plot_thresh_tmp]):
                    print j, line
                    writer.writerow(line)            

    return



###############################################################################
def plot_vals(args, im_path, data_all_classes, outpkl, figname, plot_thresh,
              verbose=False):
    '''
    Iterate through files and plot on valid_im
    see modular_sliding_window.py
    each line of input data is: [file, xmin, ymin, xmax, ymax]
    outlist has format: [xmin, ymin, xmax, ymax, filename, file_v, prob, 
                         color0, color1, color2, labeltmp, labeltmp_full] = b
    '''
    

    ##########################################
    # import slice_im, convert, post_process scripts
    sys.path.append(os.path.join(args.yolt_dir, 'scripts'))
    import yolt_post_process

    
    ####################################f######
    # APPEARANCE SETTINGS
    # COLORMAP
    if len(args.object_labels) > 1:
        # boat/plane colormap    
        colormap = [(255, 0,   0),
                    (0,   255, 0),
                    (0,   0,   255),
                    (255, 255, 0),
                    (0,   255, 255),
                    (255, 0,   255),
                    (0,   0,   255),
                    (127, 255, 212),
                    (72,  61,  139),
                    (255, 127, 80),
                    (199, 21,  133),
                    (255, 140, 0),
                    (0, 165, 255)] 
    else:
        # airport colormap
        colormap = [(0, 165, 255), (0, 165, 255)]
            
    # TEXT FONT
    # https://codeyarns.files.wordpress.com/2015/03/20150311_opencv_fonts.png
    font = cv2.FONT_HERSHEY_TRIPLEX  #FONT_HERSHEY_SIMPLEX 
    font_size = 0.4
    font_width = 1
    text_offset = [3, 10]          

    # add border
    # http://docs.opencv.org/3.1.0/d3/df2/tutorial_py_basic_ops.html
    # top, bottom, left, right - border width in number of pixels in corresponding directions
    border = (40, 0, 0, 200) 
    border_color = (255,255,255)
    label_font_width = 1 
    ##########################################
    
    # import vis            
    vis = cv2.imread(im_path, 1)  # color
    vis_h,vis_w = vis.shape[:2]
    #fig, ax = plt.subplots(figsize=args.figsize)
    img_mpl = vis #cv2.cvtColor(vis, cv2.COLOR_BGR2RGB
    
    if args.rotate_boxes:
        gray = cv2.cvtColor(vis,cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        #lines = cv2.HoughLines(edges,1,np.pi/180,50)

    out_list = []
    boxes = []
    boxes_nms = []
    legend_dic = {}
    
    if verbose:
        print "data_all_classes.columns:", data_all_classes.columns
    
    # group by category
    group2 = data_all_classes.groupby('Category')
    for i,(category, plot_df) in enumerate(group2):
        print "Plotting category:", category
        label_int = args.object_labels.index(category)
        color = colormap[label_int]#
        print "color:", color
        label = str(label_int)
        label_str = args.object_labels[label_int] #label_root0.split('_')[-1]
        print "label:", label, 'label_str:', label_str
        legend_dic[label_int] = (label_str, color)
        
        for index, row in plot_df.iterrows():
            #columns:Index([u'Loc_Tmp', u'Prob', u'Xmin', u'Ymin', 
            #                u'Xmax', u'Ymax', u'Category', 
            #                u'Image_Root_Plus_XY', u'Image_Root', u'Slice_XY', 
            #                u'Upper', u'Left', u'Height', u'Width', u'Pad', 
            #                u'Image_Path']
            
            filename, prob0, xmin0, ymin0, xmax0, ymax0, category, image_root_pxy, \
                image_root, slice_xy, upper, left, sliceHeight, sliceWidth, \
                pad, im_width, im_height, image_path, xmin, xmax, ymin, ymax = row
            #filename, prob0, xmin0, ymin0, xmax0, ymax0, category, image_root_pxy, \
            #    image_root, slice_xy, upper, left, sliceHeight, sliceWidth, \
            #    pad, image_path = row            
            prob = float(prob0)
            if prob >= plot_thresh:
                
                                    
                # skip if near edge (set edge_buffer_valid < 0 to skip)
                if args.edge_buffer_valid > 0:
                    if ((float(xmin0) < args.edge_buffer_valid) or 
                        (float(xmax0) > (sliceWidth - args.edge_buffer_valid)) or                      
                        (float(ymin0) < args.edge_buffer_valid) or 
                        (float(ymax0) > (sliceHeight - args.edge_buffer_valid)) ):
                        print "Too close to edge, skipping", row, "..."
                        continue
                
#                # below is accomplished when df is created
#                # set min, max x and y for each box, shifted for appropriate
#                #   padding                
#                xmin = max(0, int(round(float(xmin0)))+left - pad) 
#                xmax = min(vis_w, int(round(float(xmax0)))+left - pad)
#                ymin = max(0, int(round(float(ymin0)))+upper - pad)
#                ymax = min(vis_h, int(round(float(ymax0)))+upper - pad)
#                
#                # rescale output box size if desired, might want to do this
#                #    if the training boxes were the wrong size
#                if args.valid_box_rescale_frac != 1.0:
#                    dl = args.valid_box_rescale_frac
#                    xmid, ymid = np.mean([xmin, xmax]), np.mean([ymin, ymax])
#                    dx = dl*(xmax - xmin) / 2
#                    dy = dl*(ymax - ymin) / 2
#                    x0 = xmid - dx
#                    x1 = xmid + dx
#                    y0 = ymid - dy
#                    y1 = ymid + dy
#                    xmin, xmax, ymin, ymax = x0, x1, y0, y1
#
                # set coords
                coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], 
                          [xmin, ymax]]
                if args.rotate_boxes:
                    coords = yolt_post_process.rotate_box(xmin, xmax, ymin, 
                                                          ymax, canny_edges)                    
    
                out_row = [xmin, ymin, xmax, ymax, coords, filename, category, prob, 
                           color[0], color[1], color[2], label, label_str]

                #out_list.append(out_row)
                boxes.append(out_row)
                # add to plot?
                # could add a function to scale thickness with prob
                if args.nms_overlap <= 0:
                    
                    if not args.rotate_boxes:
                        cv2.rectangle(img_mpl, (xmin, ymin), (xmax, ymax), 
                                      (color), args.plot_line_thickness)   
                        # plot text
                        if args.plot_names:
                            cv2.putText(img_mpl, label, (int(xmin)
                                +text_offset[0], int(ymin)+text_offset[1]), 
                                font, font_size, color, font_width, 
                                cv2.CV_AA)#, cv2.LINE_AA)
                    else:
                        # plot rotated rect
                        coords1 = coords.reshape((-1,1,2))
                        cv2.polylines(img_mpl, [coords1], True, color, 
                                      thickness=args.plot_line_thickness)
                                
                        
    # apply non-max-suppresion on total 
    if args.nms_overlap > 0:
        boxes_nms, boxes_tot_nms, nms_idxs = non_max_suppression(boxes, 
                                                            args.nms_overlap)
        out_list = boxes_tot_nms
        # plot
        #for itmp,b in enumerate(boxes_nms):
        #    [xmin, ymin, xmax, ymax] = b
        #    color = out_list[itmp][-1]
        #    cv2. rectangle(img_mpl, (xmin, ymin), (xmax, ymax), (color), thickness)
        for itmp,b in enumerate(boxes_tot_nms):
            [xmin, ymin, xmax, ymax, coords, filename, v, prob, color0, 
                             color1, color2, labeltmp, labeltmp_full] = b
            color = (int(color0), int(color1), int(color2))
            
            if not args.rotate_boxes:
                cv2.rectangle(img_mpl, (int(xmin), int(ymin)), (int(xmax), 
                                        int(ymax)), (color), 
                                        args.plot_line_thickness)
                if args.plot_names:
                    cv2.putText(img_mpl, labeltmp, (int(xmin)+text_offset[0], 
                                                    int(ymin)+text_offset[1]), 
                                                    font, font_size, color, 
                                                    font_width, 
                                                    cv2.CV_AA)#, cv2.LINE_AA)

            else:
                # plot rotated rect
                coords1 = coords.reshape((-1,1,2))
                cv2.polylines(img_mpl, [coords1], True, color, 
                              thickness=args.plot_line_thickness)                    
    else:
        out_list = boxes
        
    # add extra classifier pickle, if desired
    if args.extra_pkl:
        labeltmp = 'airport'
        extra_idx = len(colormap) - 1
        [out_list_ex, boxes_ex, boxes_nms_ex] \
                    = pickle.load(open(args.extra_pkl, 'rb'))
        for itmp,b in enumerate(out_list_ex):
            [xmin, ymin, xmax, ymax, filename, v, prob,color0,color1,color2] = b
            color_ex = colormap[extra_idx]
            cv2.rectangle(img_mpl, (int(xmin), int(ymin)), (int(xmax), 
                                    int(ymax)), (color_ex), 
                                    2*args.plot_line_thickness)
            if args.plot_names:
                cv2.putText(img_mpl, labeltmp, (int(xmin)+text_offset[0], 
                                                int(ymin)+text_offset[1]), 
                                                font, font_size, color_ex, 
                                                font_width,  
                                                cv2.CV_AA)#cv2.LINE_AA)

                legend_dic[extra_idx] = (labeltmp, color_ex)

    # add legend and border, if desired
    if args.valid_make_legend_and_title.upper() == 'TRUE':

        # add border
        # http://docs.opencv.org/3.1.0/d3/df2/tutorial_py_basic_ops.html
        # top, bottom, left, right - border width in number of pixels in 
        # corresponding directions
        img_mpl = cv2.copyMakeBorder(img_mpl, border[0], border[1], border[2], 
                                     border[3], 
                                     cv2.BORDER_CONSTANT,value=border_color)

        xpos = img_mpl.shape[1] - border[3] + 15
        ydiff = border[0]
        for itmp, k in enumerate(sorted(legend_dic.keys())):
            labelt, colort = legend_dic[k]                             
            text = '- ' + labelt #str(k) + ': ' + labelt
            ypos = border[0] + (2+itmp) * ydiff
            cv2.putText(img_mpl, text, (int(xpos), int(ypos)), font, 
                        1.5*font_size, colort, label_font_width, 
                        cv2.CV_AA)#cv2.LINE_AA)
    
        # legend box
        cv2.rectangle(img_mpl, (xpos-5, 2*border[0]), (img_mpl.shape[1]-10, 
                      ypos+int(0.75*ydiff)), (0,0,0), label_font_width)   
                                          
        # title                                  
        title_pos = (border[0], int(border[0]*0.66))
        title = figname.split('/')[-1].split('_')[0] + ':  Plot Threshold = ' \
                        + str(plot_thresh) # + ': thresh=' + str(plot_thresh)
        cv2.putText(img_mpl, title, title_pos, font, 1.7*font_size, (0,0,0), 
                    label_font_width,  
                    cv2.CV_AA)#cv2.LINE_AA)

    print "Saving to files", outpkl, figname, "..."
 
    if len(outpkl) > 0:
        pickle.dump([out_list, boxes, boxes_nms], open(outpkl, 'wb'), protocol=2)
    
    if len(figname) > 0:   
        # save high resolution
        #plt.savefig(figname, dpi=args.dpi)  
        img_mpl_out = img_mpl #= cv2.cvtColor(img_mpl, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(figname, img_mpl_out)
        # compress?
        cv2.imwrite(figname, img_mpl_out,  [cv2.IMWRITE_PNG_COMPRESSION, args.valid_im_compression_level])

        if args.show_valid_plots:
            #plt.show()
            cmd = 'eog ' + figname + '&'
            os.system(cmd)
   
    return out_list
    
    
###############################################################################
def non_max_suppression(boxes, overlapThresh):
    '''
    Non max suppression (assume boxes = [[xmin, ymin, xmax, ymax, ...\
                             sometiems extra cols are: filename, v, prob, color]]
    # http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # Malisiewicz et al.
    see modular_sliding_window.py, functions non_max_suppression, \
            non_max_supression_rot
    '''
    
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], [], []
    
    boxes_tot = boxes#np.asarray(boxes)
    boxes = np.asarray([b[:4] for b in boxes])
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes    
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    outboxes = boxes[pick].astype("int")
    #outboxes_tot = boxes_tot[pick]
    outboxes_tot = [boxes_tot[itmp] for itmp in pick]
    
    return outboxes, outboxes_tot, pick

###############################################################################
def building_polys_to_csv(image_name, building_name, coords, conf=0, 
                          asint=True, rotate_boxe=True):
    '''
    for spacenet data
    coords should have format [[x0, y0], [x1, y1], ... ]
    Outfile should have format: 
            ImageId,BuildingId,PolygonWKT_Pix,Confidence
            https://gis.stackexchange.com/questions/109327/convert-list-of-coordinates-to-ogrgeometry-or-wkt
            https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
    '''
                
    if asint:
        coords = np.array(coords).astype(int)
            
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in coords:
        ring.AddPoint(coord[0], coord[1])
    # add first point to close polygon
    ring.AddPoint(coords[0][0], coords[0][1])
    
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    wktpoly = poly.ExportToWkt()
    
    row = [image_name, building_name, wktpoly, conf]
    return row
    

            
###############################################################################
###############################################################################
def main():
    
    print "Run YOLT2"
    args = init_args()
    os.chdir(args.yolt_dir)
    
    ##########################
    # import slice_im, convert, post_process scripts
    sys.path.append(os.path.join(args.yolt_dir, 'scripts'))
    import slice_im
    import yolt_post_process
    #import convert
    ##########################
    
    print "Date string:", args.date_string
    
    if args.mode == 'compile':
        print "Creating label images..."
        make_label_images(args.label_image_dir, new_labels=args.object_labels)
        print "Recompiling yolt..."
        recompile_darknet(args.yolt_dir) 
        return

    # make dirs
    os.mkdir(args.results_dir)
    os.mkdir(args.log_dir)

    # copy this file (yolt2.py) as well as config, plot file to results_dir
    shutil.copy2(args.this_file, args.log_dir)
    shutil.copy2(args.plot_file, args.log_dir)
    print "cfg_file:", args.cfg_file_in
    print "log_dir:", args.log_dir
    shutil.copy2(args.cfg_file_in, args.log_dir)
    # save labels to log_dir
    #pickle.dump(args.object_labels, open(args.log_dir \
    #                                    + 'labels_list.pkl', 'wb'), protocol=2)
    labels_log_file = os.path.join(args.log_dir, 'labels_list.txt')
    with open (labels_log_file, "wb") as fp:
        for ob in args.object_labels:
           fp.write(ob+"\n")

    print "Updating yolt params in files..."
    replace_yolt_vals(args)    
    # print a few values...
    print "Final output layer size:", args.final_output
    print "side size:", args.side
    print "batch_size:", args.batch_size
    print "subdivisions:", args.subdivisions
    
    # set out command       
    outcmd = yolt_command(args)
    print "\noutcmd:", outcmd
    #os.system('echo ' + outcmd + ' >> ' + args.log_file)

    # create log file, init to the contents in this file, and cfg file
    os.system('cat ' + args.cfg_file_tot + ' >> ' + args.log_file )      
    os.system('cat ' + args.this_file + ' >> ' + args.log_file)      
 
    args_str = '"\nArgs: ' +  str(args) + '\n"'
    print args_str  
    os.system('echo ' + args_str + ' >> ' + args.log_file)


    # run, if not valid
    if args.mode != 'valid':
        # run command
        t0 = time.time()
        os.system(outcmd)       #run_cmd(outcmd)
        #run_cmd(outcmd)
        #tot_cmd = outcmd + ' 2>&1 | tee -a ' + log_file
        #os.system(tot_cmd)
        #subprocess.call(outcmd + ' 2|tee ' + log_file + ' 1>>2', shell=True)
        
        cmd_time_str = '"Length of time to run command: ' +  outcmd + ' ' \
                        + str(time.time() - t0) + ' seconds\n"'
        print cmd_time_str  
        #with open(log_file, "a") as text_file:
        #    text_file.write('\n' + cmd_time_str + '\n')
        os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)
        return
    

    # need to split file for valid first, then run command
    elif args.mode == 'valid':
        
        t00 = time.time()
        run_valid(args)
        cmd_time_str = '"Length of time to run valid' + ' ' \
                        + str(time.time() - t00) + ' seconds\n"'
        print cmd_time_str  
        os.system('echo ' + cmd_time_str + ' >> ' + args.log_file)
        #with open(log_file, "w") as text_file:
        #    text_file.write(cmd_time_str+ '\n\n')  
    
    
    print "\nNo honeymoon. This is business."
    return
        
###############################################################################
###############################################################################    
if __name__ == "__main__":
    
    print "\nPermit me to introduce myself...\n\n" \
            "Well, I’m glad we got that out of the way.\n\n\n\n"
    main()
    
###############################################################################
###############################################################################
