# YOLT #

## You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery

![Alt text](/test_images/header.jpg?raw=true "")


____
### As of 24 October 2018 YOLT has been superceded by [SIMRDWN](https://github.com/CosmiQ/simrdwn) 
____


YOLT is an extension of the [YOLO v2](https://pjreddie.com/darknet/yolov2/) framework that can evaluate satellite images of arbitrary size, and runs at ~50 frames per second.  Current applications include vechicle detection (cars, airplanes, boats), building detection, and airport detection.

The YOLT code alters a number of the files in src/*.c to allow further functionality.  We also built a python wrapper around the C functions to improve flexibility.  We utililize the default data format of YOLO, which places images and labels in different directories.  For example: 

    /data/images/train1.tif
    /data/labels/train1.txt

Each line of the train1.txt file has the format

    <object-class> <x> <y> <width> <height>

Where x, y, width, and height are relative to the image's width and height. Labels can be created with [LabelImg](https://github.com/tzutalin/labelImg), and converted to the appropriate format with the /yolt/scripts/convert.py script.  


### For more information, see:

1. [arXiv paper: You Only Look Twice: Rapid Multi-Scale Object Detection In Satellite Imagery](https://arxiv.org/abs/1805.09512)

2. [Blog1: You Only Look Twice — Multi-Scale Object Detection in Satellite Imagery With Convolutional Neural Networks (Part I)](https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-38dad1cf7571)

3. [Blog2: You Only Look Twice (Part II) — Vehicle and Infrastructure Detection in Satellite Imagery](https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-34f72f659588)

4. [Blog3: Building Extraction with YOLT2 and SpaceNet Data](https://medium.com/the-downlinq/building-extraction-with-yolt2-and-spacenet-data-a926f9ffac4f)

5. [Blog4: Car Localization and Counting with Overhead Imagery, an Interactive Exploration
](https://medium.com/the-downlinq/car-localization-and-counting-with-overhead-imagery-an-interactive-exploration-9d5a029a596b)

6. [Blog5: The Satellite Utility Manifold; Object Detection Accuracy as a Function of Image Resolution
](https://medium.com/the-downlinq/the-satellite-utility-manifold-object-detection-accuracy-as-a-function-of-image-resolution-ebb982310e8c)

7. [Blog6: Panchromatic to Multispectral: Object Detection Performance as a Function of Imaging Bands](https://medium.com/the-downlinq/panchromatic-to-multispectral-object-detection-performance-as-a-function-of-imaging-bands-51ecaaa3dc56)

---

## Installation #

The following has been tested on Ubuntu 16.04.2

1. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

2. Build docker file

        nvidia-docker build -t yolt yolt_docker_name /path_to_yolt/docker

3. Launch the docker container

        nvidia-docker run -it -v /raid:/raid yolt_docker_name
        # '/raid' is the root directory of your machine, which will
        # be shared with the docker container

4. Run Makefile

        cd /path_to_yolt/
        make clean
        make
        
---

## Execution #

Commands should be executed within the docker file.  To run the container (with name yolt_train0):

    nvidia-docker run -it -v --name yolt_train0 yolt_docker_name


### HELP
    cd /path_to_yolt/scripts
    python yolt2.py --help


### TRAIN (gpu_machine)


    # e.g. train boats and planes
    cd /path_to_yolt/scripts
    python yolt2.py \
        --mode train \
        --outname 3class_boat_plane \
        --object_labels_str  boat,boat_harbor,airplane \
        --cfg_file ave_standard.cfg  \
        --nbands 3 \
        --train_images_list_file boat_airplane_all.txt \
        --single_gpu_machine 0 \
        --keep_valid_slices False \
        --max_batches 60000 \
        --gpu 0

### VALIDATE (gpu_machine)

    # e.g. test on boats, cars, and airplanes
    cd /path_to_yolt/scripts
    python yolt2.py \
        --mode valid \
        --outname qgis_labels_all_boats_planes_cars_buffer \
        --object_labels_str airplane,airport,boat,boat_harbor,car \
        --cfg_file ave_standard.cfg \
        --valid_weight_dir train_cowc_cars_qgis_boats_planes_cfg=ave_26x26_2017_11_28_23-11-36 \
        --weight_file ave_standard_30000_tmp.weights \
        --valid_testims_dir qgis_validation/all \
        --keep_valid_slices False \
        --valid_make_pngs True \
        --valid_make_legend_and_title False \
        --edge_buffer_valid 1 \
        --valid_box_rescale_frac 1 \
        --plot_thresh_str 0.4 \
        --slice_sizes_str 416 \
        --slice_overlap 0.2 \
        --gpu 2


---

## To Do #
1. Include train/test example
2. Upload data preparation scripts
3. Describe multispectral data handling
4. Describle initial results with YOLO v3
5. Describe improve labeling methods


---

_If you plan on using YOLT in your work, please consider citing [YOLO](https://arxiv.org/abs/1612.08242) and [YOLT](https://arxiv.org/abs/1805.09512)_
