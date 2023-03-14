# ML_DefectDetection
Machine learning models to detect topological defects in biological tissues. A full description of the rationale and methodology can be found online at [[insert arxiv url]].

This repository contains a pre-trained convolutional neural network (CNN) model, created using TensorFlow, for detecting nematic defects in 2D biological tissues (found in 'SavedModel'). It can be run using 'detect_defects'. Each input file should contain rows of 3 columns, with each row corresponding to a different constituent/cell and the 3 columns corresponding to x position of the cell's centre of mass (CoM), its y position and the orientation of the cell's long axis direction respectively. Input files should be saved in a folder called 'CellFiles' (or the variable 'data_filepath' should be updated). The x and y coordinates of the cell's CoM should be appropriately scaled such that the average width of defects in the system corrsponds to 2 length units (e.g. if cells are 20um across and defect cores are typically 4 cells in width then 40um = 1 length unit in saved files). Detected defects will be saved in folders called 'PosDefects' and 'NegDefects', so folders with these names should be created at the desired location (with the desired location set by the variable 'save_filepath'). Alternatively, defects can be save in folders other than 'PosDefects' and 'NegDefects' by updating the relevent filepath in the function 'SaveDefects' in functions.py.

A folder with example input data, generated using an Active Vertex Model (AVM) [1], can be found at 'CellFiles'.

In addition to using the trained model, different models can be built, trained and tested using 'train_and_test_model', which uses 'nn_inputs' and correpsonding 'nn_labels' to train and test the model. These preprocessed inputs were obtained from AVM simulations.

[1] Andrew Killeen, Thibault Bertrand, Chiu Fan Lee, Polar Fluctuations Lead to Extensile Nematic Behavior in Confluent Tissues, Physical Review Letters, 128, 078001 (2022).

