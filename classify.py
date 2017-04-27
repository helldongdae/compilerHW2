#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time

import caffe

# Reads in the answers text
f = open("ans.txt", "r")
ans = f.readlines()

def main(argv):
# path of pycaffe
    pycaffe_dir = os.path.dirname(__file__)

# argument parser. Not necessary
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--input_file",
	default='photo.jpg',
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "--output_file",
	default='result',
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "Fdeploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "finetune.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
# whien store_false or store_true is used for default, they have True and False values respectively
        action='store_false',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='224,224',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=' ',
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )

# Parse arguments
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    
# Channel swap for RGB to BGR
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

# Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

# Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('npy'):
        #print("Loading file: %s" % args.input_file)
        inputs = np.load(args.input_file)
    elif os.path.isdir(args.input_file):
        #print("Loading folder: %s" % args.input_file)
        inputs =[caffe.io.load_image(im_f)
                 for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
    else:
        #print("Loading file: %s" % args.input_file)
	args.input_file = "uploads/" + args.input_file;
        inputs = [caffe.io.load_image(args.input_file)]

    #print("Classifying %d inputs." % len(inputs))

# Settings for GPU or CPU computation
    if args.gpu:
        caffe.set_mode_gpu()
        #print("GPU mode")
    else:
        caffe.set_mode_cpu()
        #print("CPU mode")

# Classify.
    start = time.time()
    predictions = classifier.predict(inputs, not args.center_only)
    #print("Done in %.2f s." % (time.time() - start))

# Save (Non neccessary)
    #print("Saving results into %s" % args.output_file)
    lst = []
    for i in range(30):
	lst.append((predictions[0][i], i))
    lst.sort(reverse=True)
    print ans[lst[0][1]].strip() +" "+ ans[lst[1][1]].strip() +" "+ ans[lst[2][1]].strip() +" "+ ans[lst[3][1]].strip() +" "+ ans[lst[4][1]].strip() +" "+ ans[lst[5][1]].strip()
    #print predictions[0]

if __name__ == '__main__':
    main(sys.argv)
