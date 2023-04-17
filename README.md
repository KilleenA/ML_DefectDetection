# ML_DefectDetection
Machine learning models to detect topological defects in biological tissues. A full description of the rationale and methodology can be found online at https://arxiv.org/abs/2303.08166.

This repository contains a pre-trained convolutional neural network (CNN) model, created using TensorFlow, for detecting nematic defects in 2D biological tissues (found in 'SavedModel'). It can be run using 'detect_defects'. Each input file should contain rows of 3 columns, with each row corresponding to a different constituent/cell and the 3 columns corresponding to x position of the cell's centre of mass (CoM), its y position and the orientation of the cell's long axis direction respectively. The x and y coordinates of the cell's CoM should be appropriately scaled such that one length unit corresponds to the characteristic length of one cell (e.g. if cells are ~20um in width then 20um = 1 length unit in saved files). 

In detect_defects the user must set the location of the input data (data_filepath) and the desired location of the detected defect coordinates (save_filepath, posdef_path and negdef_path). The grid spacing must also be chosen such that defect cores are properly captured within 'regions of interest'. grid_spacing should be chosen such that the defect core can be captured within a window with sides of length ~9*grid_spacing

A folder with example input data, generated using an Active Vertex Model (AVM) [1], can be found at 'CellFiles'.

In addition to using the trained model, different models can be built, trained and tested using 'train_and_test_model', which uses 'nn_inputs' and correpsonding 'nn_labels' to train and test the model. These preprocessed inputs were obtained from AVM simulations.

The detected defects can also be plotted, along with the coordinates of the cells' centre of mass and orientation of their long-axes, using the 'defect_plotting.gnu'. This requires the installation of plotting software gnuplot (freely available at: https://sourceforge.net/projects/gnuplot/).

[1] Andrew Killeen, Thibault Bertrand, Chiu Fan Lee, Polar Fluctuations Lead to Extensile Nematic Behavior in Confluent Tissues, Physical Review Letters, 128, 078001 (2022).
