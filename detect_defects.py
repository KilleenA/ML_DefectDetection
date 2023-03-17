#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for using model and detecting defects. Input data files should be saved in 
a folder called 'CellFiles' (or 'data_filepath' should appropriately changed), with each 
file being a as a comma separated text file where each line contains the x, y
and orientation angle of each cell. 

x and y coordinates should be scaled such that one length unit corresponds to the 
characteristic length of one cell. 

grid_spacing should then be chosen such that the defect core can be captured
by a window with sides of length ~9*grid_spacing

save_filepath, posdef_path and negdef_path should be changed to desired destinations
of defect coordinate files
"""
import os
import functions as f
from tensorflow import keras

grid_space = 0.2                            #Choose spacing of interpolation grid
data_filepath = './CellFiles'               #Location of experimental data files
save_filepath = './DefectFiles'             #Path to where defect folders will be located
posdef_path = save_filepath + '/PosDefects' #Location of +1/2 defect files
negdef_path = save_filepath + '/NegDefects' #Location of -1/2 defect files
angles = True #Whether to save the detected defects orientation along with position 

#Load CNN model
model_filepath ='./SavedModel'
model = keras.models.load_model(model_filepath)

#Load experimental data
files = [file for file in sorted(os.listdir(data_filepath))]

#Detect defects
for i,file in enumerate(files):
    file_w_path = os.path.join(data_filepath, file)
    pos_defs,neg_defs = f.DetectDefects(file_w_path,model,grid_space,angles)
    f.SaveDefects(posdef_path,negdef_path,pos_defs,neg_defs,i)
       
    if i%20 == 0:
        print('Detected defects in '+str(i)+' files')
    