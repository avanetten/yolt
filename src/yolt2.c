// IN YOLOV2 DETECTION IS CALLED WITH ./DARKNET DETECT ..., WHICH CALLS
// TRAIN DETECTOR.C.  THEREFORE THIS FILE IS A HIGHLY MODIFIED VERSION 
// OF DETECTOR.C

#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

// for list parsing
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
	
// Split string into list
// http://stackoverflow.com/questions/9210528/split-string-with-delimiters-in-c
char** str_split2(char* a_str, const char a_delim)
{
    char** result    = 0;
    size_t count     = 0;
    char* tmp        = a_str;
    char* last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    /* Count how many elements will be extracted. */
    while (*tmp)
    {
        if (a_delim == *tmp)
        {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (a_str + strlen(a_str) - 1);

    /* Add space for terminating null string so caller
       knows where the list of returned strings ends. */
    count++;

    result = malloc(sizeof(char*) * count);

    if (result)
    {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);

        while (token)
        {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        assert(idx == count - 1);
        *(result + idx) = 0;
    }

    return result;
}

image rgb_from_multiband2(image im)
{
	//ave_edit
	//take first three bands from multiband
	int nbands_out = 3;
    image im_out = make_image(im.w, im.h, nbands_out);
    int x,y,k;
    for(k = 0; k < nbands_out; ++k){
        for(y = 0; y < im.h; ++y){
            for(x = 0; x < im.w; ++x){
                float val = get_pixel(im, x,y,k);
                set_pixel(im_out, x, y, k, val);
            }
        }
    }
	return im_out;
}

void save_image_ave2(image im, char *out_dir, const char *name)
{
    char buff[256];
    //char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
	//sprintf(buff, "%s.png", name);
	fprintf(stderr, "out_name: %s/%s.png", out_dir, name);
    sprintf(buff, "%s/%s.png", out_dir, name);	
    unsigned char *data = calloc(im.w*im.h*im.c, sizeof(char));
    int i,k;
    for(k = 0; k < im.c; ++k){
        for(i = 0; i < im.w*im.h; ++i){
            data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
        }
    }
    int success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
    free(data);
    if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
}

void train_yolt2(char *cfgfile, char *weightfile, char *train_images, char *results_dir, int nbands, char *loss_file)//, int *gpus, int ngpus)//, int reset_seen)
{
	int reset_seen = 1;
	int ngpus = 1;
	int *gpus;
	
    srand(time(0));
    //data_seed = time(0);
    char *base = basecfg(cfgfile);
    fprintf(stderr, "Base: %s\n", base);
    float avg_loss = -1;
	
	// set loss file
	FILE *lossfile;
	//if(loss_file){
	//	lossfile = fopen(loss_file, "a");
	//}
	lossfile = fopen(loss_file, "a");
    if (lossfile == NULL) {
       fprintf(stderr, "Couldn't open lossfile for appending.\n");
       exit(0);
    }
	fprintf(lossfile, "%s,%s,%s,%s\n", "Batch_Num", "BatchSize", "N_Train_Ims", "Loss");
    fclose(lossfile);
    fprintf(stderr, "%s,%s,%s,%s\n", "Batch_Num", "BatchSize", "N_Train_Ims", "Loss");
    
    network net = parse_network_cfg(cfgfile);
    network *nets = calloc(ngpus, sizeof(network));
	int i;
	int imgs;
	
    if(ngpus == 1){
		////////////////
		// single gpu
	    //network net = parse_network_cfg(cfgfile);
	    if(weightfile){
	        load_weights(&net, weightfile);
	    }

	    //if using pretrained network, reset net.seen
		if(reset_seen){
			*net.seen = 0;
		}

	    imgs = net.batch*net.subdivisions;
	    i = *net.seen/imgs;
		////////////////////				
    } else {
		/////////////////////
		// multi gpu
	    network *nets = calloc(ngpus, sizeof(network));

	    srand(time(0));
	    int seed = rand();
	    int i;
	    for(i = 0; i < ngpus; ++i){
	        srand(seed);
#ifdef GPU
	        cuda_set_device(gpus[i]);
#endif
	        nets[i] = parse_network_cfg(cfgfile);
	        if(weightfile){
	            load_weights(&nets[i], weightfile);
	        }
	        if(reset_seen) *nets[i].seen = 0;
	        nets[i].learning_rate *= ngpus;
	    }
	    srand(time(0));
	    network net = nets[0];

	    imgs = net.batch * net.subdivisions * ngpus;
	////////////////////		
    }
	
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    fprintf(stderr, "\n\n\n\n\n\nNum images = %d,\ni= %d\n", imgs, i);
    data train, buffer;

    layer l = net.layers[net.n - 1];

    //int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
	args.c = nbands;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 8;
	//args.load_data_verbose = 1;  // Set switch in data.c.load_data_detection()

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
	
    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
	clock_t begin;
	clock_t end;
	
	fprintf(stderr, "N ims: %d\n", N);
	int stop_count = net.max_batches;
	fprintf(stderr, "Num iters: %d\n", stop_count);
	begin=clock();
	
    //while(i*imgs < N*120){
	int count = 0;
	int j = 0;
    while(get_current_batch(net) < net.max_batches){
		j += 1;
        if(l.random && count++%10 == 0){
            fprintf(stderr, "Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+100 > net.max_batches) dim = 544;
            //int dim = (rand() % 4 + 16) * 32;
            fprintf(stderr, "%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for(i = 0; i < ngpus; ++i){
                resize_network(nets + i, dim, dim);
            }
            net = nets[0];
        }
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);				
	
        fprintf(stderr, "Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
		fprintf(stderr, "Batch Num: %d / %d\n", j, net.max_batches);
		
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
		
		//if (loss_file){
		//	fprintf(lossfile, "%f\n", loss);
		//}
		if (i % 50 == 0){
            FILE *lossfile = fopen(loss_file, "a");
            fprintf(lossfile, "%d,%d,%d,%f\n", i, imgs, N, loss);
            fclose(lossfile);
			//fprintf(lossfile, "%d,%d,%d,%f\n", i, imgs, N, loss);
		}
		
        i = get_current_batch(net);
        fprintf(stderr, "%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        //printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        //if(i%2000==0 || (i < 1000 && i% 100 == 0)){
	    //if(i%5000==0){
	    if(i%5000==0 || (i == 1000)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d_tmp.weights", results_dir, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", results_dir, base);
    save_weights(net, buff);
	end = clock();
    fprintf(stderr, "Total Elapsed for Training: %f seconds\n", (double)(end-begin) / CLOCKS_PER_SEC);
	//if (loss_file){
	//	fclose(lossfile);
	//}
	fclose(lossfile);
}
			

void print_yolt_detections2(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_yolt2(char *cfgfile, char *weightfile, char *valid_list_loc, 
		float iou_thresh, char *names[], char *results_dir, int nbands)
{
    int *map = 0;
    network net = parse_network_cfg(cfgfile);
    //if(weightfile){
	if(strcmp(weightfile, "0") != 0){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    fprintf(stderr, "valid_list_loc: %s\n", valid_list_loc);

    srand(time(0));

    list *plist = get_paths(valid_list_loc);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    //int square = l.sqrt;
    //int side = l.side;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
		fprintf(stderr, "Output file: %s/%s.txt\n", results_dir, names[j]);		
		snprintf(buff, 1024, "%s/%s.txt", results_dir, names[j]);        
		fps[j] = fopen(buff, "w");
    }
    
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));
	
	// old
	//box *boxes = calloc(side*side*l.n, sizeof(box));	
    //float **probs = calloc(side*side*l.n, sizeof(float *));
    //for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    //float iou_thresh = .5;
    int nms = 0;
	if (iou_thresh > 0) nms=1;
	
    int nthreads = 1;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
	args.c = nbands;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
	time_t start = time(0);
	
	//https://stackoverflow.com/questions/16275444/how-to-print-time-difference-in-accuracy-of-milliseconds-and-nanoseconds
	struct timespec tstart={0,0}, tend={0,0};
    clock_gettime(CLOCK_MONOTONIC, &tstart);
	
    //time_t start, stop;
	//clock_t ticks;

    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d / %d \n", i, m);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
			
			// set id as path, not the partial path
            //char *id = basecfg(path);
			char *id = path;
		    //fprintf(stderr, "path: %s\n", path);
			
		    fprintf(stderr, "validate id: %s\n", id);
			
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;

			// from yolo_orig.c: get_detection_boxes replaces convert_yolt_detections     
            get_region_boxes(l, w, h, thresh, probs, boxes, 0, map);        
			//get_detection_boxes(l, w, h, thresh, probs, boxes, 0);
            //convert_yolt_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
			
            if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, classes, iou_thresh);
            //if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);

			// //*************
			// if(show_labels_switch==0){
			// 		        draw_detections(val[t], l.side*l.side*l.n, thresh, boxes, probs, names, 0, CLASSNUM);
			// } else{
			// 		        draw_detections(val[t], l.side*l.side*l.n, thresh, boxes, probs, names, voc_labels, CLASSNUM);
			// }
			// //*************
			
            print_yolt_detections2(fps, id, boxes, probs, l.w*l.h*l.n, classes, w, h);
								
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
	
	//time(&stop);
    clock_gettime(CLOCK_MONOTONIC, &tend);
	
	
	fprintf(stderr, "Total Detection Time: %.5f Seconds\n", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - 
           ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
	//fprintf(stderr, "Total Detection Time: %.4f Seconds\n", (double)(time(0) - start));

    //fprintf(stderr, "Total Detection Time: %.6f Seconds\n", difftime(stop, start));
}

void test_yolt2(char *cfgfile, char *weightfile, char *filename, float plot_thresh, float iou_thresh, 
		char *names[], image *voc_labels, int CLASSNUM, int nbands, char *results_dir)
{
	
	//fprintf(stderr, "Label names: %s", names);
	//int i;
	//for(i = 0; i < 80; ++i){
	//	//printf("names i %i, %s\n", i, names[i]);
	//	fprintf(stderr, "%s ", names[i]);
	//}

	int show_labels_switch=1;
    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    //detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    int nms = 0;
	if (iou_thresh > 0) nms=1;
    //float nms=.4;
    //box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    //float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    //for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
		
        //image im = load_image_color(input,0,0);
        image im = load_image(input, 0, 0, nbands);
		image im3 = rgb_from_multiband2(im); //copy_image(im);
		
		// // if using multiband...
// 		// get 3 band image
// 		if (nbands > 3){
// 			image im3 = rgb_from_multiband2(im);
// 		}

        image sized = resize_image(im, net.w, net.h);
        layer l = net.layers[net.n-1];

        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
		fprintf(stderr, "plot thresh: %f\n", plot_thresh);
        get_region_boxes(l, 1, 1, plot_thresh, probs, boxes, 0, 0);
		//fprintf(stderr, "get_region_boxes() successful\n");
        if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, iou_thresh);
		//fprintf(stderr, "do_nms_sort() successful\n");
        draw_detections(im3, l.w*l.h*l.n, plot_thresh, boxes, probs, names, alphabet, l.classes);
        save_image_ave2(im3, results_dir, "predictions");
        //save_image(im3, "predictions");
        show_image(im3, "predictions");
        //draw_detections(im, l.w*l.h*l.n, plot_thresh, boxes, probs, names, alphabet, l.classes);
        //save_image(im, "predictions");
        //show_image(im, "predictions");
		//fprintf(stderr, "draw_detections() successful\n");

        free_image(im);
		free_image(im3);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}		
		
void run_yolt2(int argc, char **argv)
{
	
	if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
	
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

	// turn names_str into names_list
	fprintf(stderr, "\nRun YOLT.C...\n");
	fprintf(stderr, "Plot Probablility Threshold: %f\n", plot_thresh);
	fprintf(stderr, "Label_str: %s\n", names_str);
	fprintf(stderr, "len(names): %i\n", len_names);
	fprintf(stderr, "num channels: %i\n", nbands);

	char **names;
	if(len_names > 0){
		names = str_split2(names_str, ',');
		fprintf(stderr, "Len names %i\n", len_names);
		//printf("names: %s", names);
		int i;
	    for(i = 0; i < len_names; ++i){
			char *ni = names[i];
			printf("label i: %i, %s\n", i, ni);
		}
		//int len_names = sizeof(*names) / sizeof(*names[0]);
		//int len_names = sizeof(names) / sizeof(*names);
		//fprintf(stderr, "Load Network:\n");
	}

    if(0==strcmp(argv[2], "test")){
		//load  labels images
		image voc_labels[len_names];
	    int i;
	    for(i = 0; i < len_names; ++i){
	        char buff[256];
	        //sprintf(buff, "data/labels/%s.png", names[i]);
			//fprintf(stderr, "i, label[i] %i %s\n", i, names[i]);
	        sprintf(buff, "data/category_label_images/%s.png", names[i]);
	        voc_labels[i] = load_image_color(buff, 0, 0);
	    }
		test_yolt2(cfg, weights, test_filename, plot_thresh, nms_thresh, names, voc_labels, len_names, 
				  nbands, results_dir);
    } 
    else if(0==strcmp(argv[2], "train")) train_yolt2(cfg, weights, train_images, results_dir, nbands, loss_file);
    else if(0==strcmp(argv[2], "valid")) validate_yolt2(cfg, weights, valid_list_loc, 
													   nms_thresh, names, results_dir, nbands);
}
