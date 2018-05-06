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
            # print(os.path.join(directory, filename))
            f.write(filename + " " + label + '\n')
            

"""
Labels files in a directory by a label identifier on their names 
"""
def label_files_by_filename(directory, output):
    mode = 'a' if os.path.isfile(output) else 'w'
    with open(output, mode) as f:
        for filename in os.listdir(directory):
            # print(os.path.join(directory, filename))
            file_label = filename.split("_")[1]
            if file_label == 'B':
                label = 0
            elif file_label == 'M':
                label = 1
            f.write(filename + " " + str(label) + '\n')
			
			
if __name__ == "__main__":        
    # Open a file to write the images and their labels on it
    output = '<path_to_output_file>/output_file.txt'
    directory = '<path_to_directory_dataset>'
    label = str(1)
    
    label_files(directory=directory, label=label, output=output)
    print 'Finished'
