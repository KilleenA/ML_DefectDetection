#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for using model and detecting defects. Input data files should be saved in 
a folder called 'CellFiles' (or 'data_filepath' should appropriately adapted), with each 
file being a as a comma separated text file where each line contains the x, y
and orientation angle of each cell. x and y coordinates should be scaled such 
that defects are approxiately two length units in width. Detected defects will be
saved in folders called 'PosDefects' and 'NegDefects' at 'save_filepath' 
"""
import os
import functions as f
from tensorflow import keras

#Load CNN model
model_filepath ='./SavedModel'
model = keras.models.load_model(model_filepath)

#Load experimental data
data_filepath = './CellFiles'
files = [file for file in sorted(os.listdir(data_filepath))]

#Path to where 'PosDefects' and 'NegDefects' will be located
save_filepath = './DefectFiles'

angles = True #Whether to save the detected defects orientation along with position 

#Detect defects
for i,file in enumerate(files):
    file_w_path = os.path.join(data_filepath, file)
    pos_defs,neg_defs = f.DetectDefects(file_w_path,model,angles)
    f.SaveDefects(save_filepath,pos_defs,neg_defs,i)
       
    if i%20 == 0:
        print('Detected defects in '+str(i)+' files')
    