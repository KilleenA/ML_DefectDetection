#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for training a CNN model, parameters are currently set to values used in 
final model.
"""
import numpy as np
import functions as f
import tensorflow as tf
import matplotlib.pyplot as plt

runs = 50                                  #Number of training runs to perform
no_of_epochs = 20                          #Number of epochs to train for
bs = 64                                    #Batch size for SGD batches
lr = 0.05                                  #Initial learning rate
conv_layers = 2                            #Number of convolution layers
features = [32,32]                         #Number of features in each layer
win_size = [6,3]                           #Window size of each layer
dense_layers = 1                           #Number of dense layers
dense_size = [100]                         #Number of neurons in dense layer
dropout = 0.5                              #% of neurons to drop out in training
initializer = 'glorot_normal'              #Distribution to initialise weights from
l2_reg = tf.keras.regularizers.l2(l2=0.01) #Amount of L2 reg in dense layer
save_model = False                         #Whether to save final trained model

#Load training and testing data
ROIs = np.loadtxt('./nn_inputs.txt')
ROI_labels = np.loadtxt('./nn_labels.txt')

#Reshape into form that can be read by CNN
ROIs = np.reshape(ROIs,(-1,9,9))

#Split data into training and test sets
train_ROIs, test_ROIs, train_labels, test_labels = f.TrainTestSplit(ROIs,ROI_labels)
#Enlarge training data set via rotations and reflections
train_ROIs,train_labels = f.EnlargeTrainingData(train_ROIs,train_labels)
#Build CNN from input parameters
model = f.BuildCNN(conv_layers,features,win_size,dense_layers,dense_size,dropout,initializer,regularizer=l2_reg,max_pool=None)
#Define parameters for triaining regime with variable learning rate
step = tf.Variable(0, trainable=False)
#Define change in lr to be halfway through training
boundaries = [int((np.size(ROI_labels,axis=0)/bs)*0.5*no_of_epochs)]
values = [lr, 0.1*lr]
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

#Initial performance metrics
val_accs = np.zeros((1,no_of_epochs))
val_loss = np.zeros((1,no_of_epochs))
accuracy =  np.zeros((runs,1));
prediction_stats =  np.zeros((runs,6));

#Loop through training runs, to get statistics on its accuracy
for i in range(runs):
    trained_model, history = f.TrainModel(train_ROIs,train_labels,model,no_of_epochs,bs,lr_sched)
    val_accs[i,:] = history.history['val_accuracy']
    val_loss[i,:] = history.history['val_loss']
    plt.plot(history.history['val_accuracy'])

    test_pred = f.MLPrediction(test_ROIs,trained_model)
    accuracy[i],prediction_stats[i,:] = f.PredictionStatistics(test_pred,test_labels)

#Assess mean test performance
test_accuracy = np.mean(accuracy)
test_stats = np.mean(prediction_stats,0)
#Assess mean training performance
training_performance = np.vstack((np.mean(val_accs,0),np.std(val_accs,0),np.mean(val_loss,0),np.std(val_loss,0)))

#Assess winding number peroformance on training data
winding_train_pred = f.WindingPrediction(train_ROIs)
winding_train_accuracy,winding_test_stats = f.PredictionStatistics(winding_train_pred,train_labels)
#Assess winding number perofrmance on test data
winding_test_pred = f.WindingPrediction(test_ROIs)
winding_test_accuracy,winding_test_stats = f.PredictionStatistics(winding_test_pred,test_labels)

print("Neural network has a mean accuracy of " + str(test_accuracy) + ", the winding number has an accuracy of " + str(winding_test_accuracy))

#Save model from final run (if desired)
if save_model:
    model_filepath = './SavedModel'
    f.SaveModel(model,model_filepath + 'InsertModelName')