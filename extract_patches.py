# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 20:58:06 2018

@author: cristian
"""
import os

from scipy import misc
from sklearn.feature_extraction import image
import sys
from labelling_images import label_files_by_filename


patch_width = patch_height = 64

         
def patches_from_image(image_path, n_patches):
    np_image_array = misc.imread(image_path)
    patches = image.extract_patches_2d(np_image_array, (patch_height, patch_width), max_patches=n_patches)
    return patches
    

def generate_patches(directory, output_directory, n_patches):
    for image_path in os.listdir(directory):
        patches = patches_from_image(directory + image_path, n_patches)
        
        image_name = image_path.split(".")[0]
        image_format = image_path.split(".")[1]
        
        print "Generating patches from image: " + image_path     
        for x in range(n_patches):
            patch = patches[x,:,:,:]
            output_patch_path = (output_directory + image_name +
                                "_patch_" + str(x) + "." + image_format) 
            misc.imsave(output_patch_path, patch)
 
 
 
 # Label both train and test set images, 0 for benign; 1 for malignant       
def label_image_sets(input_directory):
    
    parent_path = os.path.abspath(os.path.join(input_directory, os.pardir))
    
    if 'test' in input_directory:
        labels_file_path = parent_path + '/test.txt'
    elif 'train' in input_directory:
        labels_file_path = parent_path + '/train.txt'
            
    label_files_by_filename(input_directory, labels_file_path)


"""
Extracts patches for each image in both the train and testing directory and labels them.
"""	
if __name__ == "__main__":
    
    if len(sys.argv) == 4:
        
        input_train_path = sys.argv[1] + 'train_initial/'
        output_train_path = sys.argv[2] + 'train/'
        
        generate_patches(input_train_path, output_train_path, int(sys.argv[3]))
        label_image_sets(output_train_path)
        
        input_train_path = sys.argv[1] + 'test_initial/'
        output_train_path = sys.argv[2] + 'test/'
        
        generate_patches(input_train_path, output_train_path, int(sys.argv[3]))
        label_image_sets(output_train_path)
        
    else:
        print 'Error: Invalid Arguments. Default values will be used.'
        n_patches = 1000

        input_directory = ''
        
        output_directory = ''
        
        generate_patches(input_directory, output_directory, n_patches)
        
        label_image_sets(output_directory)

        

        
        
        






    