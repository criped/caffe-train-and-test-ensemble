# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:25:34 2018

@author: cristian
"""
import os


def label_files(directory, label, output):
    mode = 'a' if os.path.isfile(output) else 'w'
    with open(output, mode) as f:
        for filename in os.listdir(directory):
            f.write("{} {}\n".format(filename, label))
            

def label_files_by_filename(directory, output):
    """
    Labels files in a directory by a label identifier on their names
    """
    mode = 'a' if os.path.isfile(output) else 'w'
    with open(output, mode) as f:
        for filename in os.listdir(directory):
            # print(os.path.join(directory, filename))
            file_label = filename.split("_")[1]
            if file_label == 'B':
                label = 0
            elif file_label == 'M':
                label = 1
            else:
                raise ValueError("Tumour has to be either benign or malignant")
            f.write("{} {}\n".format(filename, label))


if __name__ == "__main__":        
    # Open a file to write the images and their labels on it
    dataset_output = '<path_to_output_file>/output_file.txt'
    dataset_directory = '<path_to_directory_dataset>'
    l = str(1)
    
    label_files(directory=dataset_output, label=l, output=dataset_directory)
    print('Finished')
