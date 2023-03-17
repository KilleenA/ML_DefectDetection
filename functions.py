# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import skimage.measure as skm
from scipy.interpolate import griddata

#Build CNN with desired architecture
def BuildCNN(conv_layers,features,win_size,dense_layers,dense_size,dropout,initializer,regularizer=None,max_pool=None):
    model = tf.keras.Sequential()
    for i in range(conv_layers):
        if i==0:
            model.add(layers.Conv2D(features[i], win_size[i], activation='relu', input_shape=(9,9,1), kernel_initializer=initializer))
        else:
            model.add(layers.Conv2D(features[i], win_size[i], activation='relu', kernel_initializer=initializer)) 
    
    model.add(layers.Flatten())
    for i in range(dense_layers):
        model.add(layers.Dense(dense_size[i], activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer))
        model.add(layers.Dropout(dropout))
        
    model.add(layers.Dense(3, activation='softmax', kernel_initializer=initializer, kernel_regularizer=regularizer))
    return model

#Build a fully-connected feedforward neural network with desired architecture
def BuildANN(input_size,dense_layers,dense_size,dropout,initializer,regularizer=None):
    model = tf.keras.Sequential()

    model.add(layers.Flatten(input_shape=input_size))
    for i in range(dense_layers):
        model.add(layers.Dense(dense_size[i], activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer))
        model.add(layers.Dropout(dropout))
        
    model.add(layers.Dense(3, activation='softmax', kernel_initializer=initializer, kernel_regularizer=regularizer))

    return model

#Train the model using SGD with a cross-entropy cost function
def TrainModel(inputs,labels,model,no_of_epochs,bs,lr):
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    history = model.fit(inputs,labels, batch_size = bs, epochs=no_of_epochs, verbose=0, validation_split = 0.1)
    trained_model = model
    
    return trained_model, history

#Save the model to the desired filepath
def SaveModel(model,filepath):
    model.save(filepath);
    return None

#Calculate the winding number around the edge of the ROIs
def WindingPrediction(inputs):
    #Initialise classification matrix
    winding_pred = np.zeros((np.size(inputs,0),3))
    #Loop through ROIs
    for i in range(np.size(inputs,0)):
        t = inputs[i,:,:]; 
        #Flatten edge of ROI into a column vector
        alpha_0 = np.concatenate((t[::-1,0],t[0,1:-1],t[:,-1],t[-1,-2:0:-1]))
        #Find the change in angle between adjacent elements
        alpha_1 = np.roll(alpha_0,1)
        #Correct any values that are not the shortest change in angle
        d_alpha = alpha_1 - alpha_0
        d_alpha[d_alpha<-np.pi/2] += np.pi
        d_alpha[d_alpha>np.pi/2] -= np.pi
        #Sum these changes to get the total winding number
        winding = round(sum(d_alpha)/(2*np.pi),1)
        
        #Update classification matrix
        if winding == -0.5:
            winding_pred[i,2] = 1
        elif winding == 0.5:
            winding_pred[i,0] = 1
        else:
            winding_pred[i,1] = 1
            
    return winding_pred

#Use model to classify inputs
def MLPrediction(inputs,model):
    ML_pred_prob = model.predict(inputs)
    #Output of model.predict will be the probability each class is correct, take
    #largest probability as the classification
    ML_pred = np.eye(3,dtype=int)[np.argmax(ML_pred_prob,axis=1)]
    
    return ML_pred

#Saving detected defects, update pos_folder and neg_folder if different locations
#are desired.
def SaveDefects(posdef_path,negdef_path,pos_defs,neg_defs,image_num):  
    #Add a larger buffer of zeros. Needs to be increased if >1e7 images are being processed
    filename = 'posdefects%06d.txt' % image_num
    np.savetxt(os.path.join(posdef_path, filename),pos_defs, delimiter=',')
    filename = 'negdefects%06d.txt' % image_num
    np.savetxt(os.path.join(negdef_path, filename),neg_defs, delimiter=',')
    return None

#Assess errors in predictions. TP = true positive, FP = false positive
#FN = false negative
def PredictionStatistics(predictions,labels):
    
    diff = predictions-labels 
    errors = np.sum(0.5*np.sum(abs(diff),axis=1))
    #Assess accuracy for +1/2 defects
    pos_TP = np.intersect1d(np.argwhere(diff[:,0]==0),np.argwhere(predictions[:,0]==1))
    pos_FP = np.argwhere(diff[:,0]==1)
    pos_FN = np.argwhere(diff[:,0]==-1)
    #Assess accuracy when no defect is present
    non_TP = np.intersect1d(np.argwhere(diff[:,1]==0),np.argwhere(predictions[:,1]==1))
    non_FP = np.argwhere(diff[:,1]==1)
    non_FN = np.argwhere(diff[:,1]==-1)
    #Assess accuracy for -1/2 defects
    neg_TP = np.intersect1d(np.argwhere(diff[:,2]==0),np.argwhere(predictions[:,2]==1))
    neg_FP = np.argwhere(diff[:,2]==1)
    neg_FN = np.argwhere(diff[:,2]==-1)

    accuracy = (len(labels)-errors)/len(labels)

    prediction_stats = np.array([len(pos_TP),len(pos_FP),len(pos_FN),len(non_TP),len(non_FP),len(non_FN),len(neg_TP),len(neg_FP),len(neg_FN)])
    
    return accuracy,prediction_stats

#Split available data into training and testing data
def TrainTestSplit(nn_inputs,nn_labels):
    train_prop = 0.9    #Proportion of inputs to use for training the model
    train_inputs = nn_inputs[:int(np.ceil(train_prop*len(nn_inputs))),:,:]
    test_inputs = nn_inputs[int(np.ceil(train_prop*len(nn_inputs))):,:,:]
    train_labels = nn_labels[:int(np.ceil(train_prop*len(nn_inputs))),:]
    test_labels = nn_labels[int(np.ceil(train_prop*len(nn_inputs))):,:]
    
    return train_inputs, test_inputs, train_labels, test_labels

#Expand training data
def EnlargeTrainingData(nn_inputs,nn_labels):
    #Rotation Enlargement
    enlarged_inputs = nn_inputs
    enlarged_labels = nn_labels
    for i in range(1,4):
        rotated_inputs = np.rot90(nn_inputs,i,axes=(1,2))
        
        if i!=2:
            rotated_inputs += np.pi/2
            rotated_inputs[rotated_inputs<-np.pi/2] += np.pi
            rotated_inputs[rotated_inputs>np.pi/2] -= np.pi
        
        enlarged_inputs = np.concatenate((enlarged_inputs,rotated_inputs))
        enlarged_labels = np.concatenate((enlarged_labels,nn_labels))
        
    #Flip Enlargement
    flipped_inputs = np.fliplr(enlarged_inputs)
    flipped_inputs = np.pi - flipped_inputs
    flipped_inputs[flipped_inputs<-np.pi/2] += np.pi
    flipped_inputs[flipped_inputs>np.pi/2] -= np.pi

    enlarged_inputs = np.concatenate((enlarged_inputs,flipped_inputs))
    enlarged_labels = np.concatenate((enlarged_labels,enlarged_labels))
    
    return enlarged_inputs, enlarged_labels

#Use a trained model to detect defects
def DetectDefects(file_w_path,model,grid_space,angles):
    #Find ROIs
    POIs,ROIs = ROIFinder(file,grid_space)
    #Classify ROIs using model
    label_prob = model.predict(ROIs,verbose=0)
    labels = np.eye(3,dtype=int)[np.argmax(label_prob,axis=1)]
    #Use labels to find coordinates of detected defects
    pos_defs = POIs[labels[:,0]==1,:];
    neg_defs = POIs[labels[:,2]==1,:];
    #Find the orientation of defects (if desired)
    if angles:
        pos_ROIs = ROIs[labels[:,0]==1,:,:];
        neg_ROIs = ROIs[labels[:,2]==1,:,:];
        
        pos_angles = DefectOrientator(pos_ROIs,0.5,grid_space)
        neg_angles = DefectOrientator(neg_ROIs,-0.5,grid_space)
    
        pos_defs = np.hstack((pos_defs,pos_angles))
        neg_defs = np.hstack((neg_defs,neg_angles))
    return pos_defs,neg_defs

#Find ROIs
def ROIFinder(file,grid_space):
    cell_data = np.loadtxt(file, delimiter = ',')
    cell_data[cell_data[:,2]<0,2] += np.pi
    
    x_max = max(cell_data[:,0])
    y_max = max(cell_data[:,1])

    offset = 4
    xg, yg = np.mgrid[0:x_max:grid_space,0:y_max:grid_space]
    #Interpolate data to grid (xg,yg) to obtain nematic field
    grid_t = GridDirectors(cell_data,xg,yg,offset)
    #Find scalar order parameter at each grid point
    S = SFinder(grid_t,offset)
    #Find centres of mass of points of low S (points of interest)
    POI_indices = POIFinder(grid_t,S,offset)
    POIs = grid_space*POI_indices
    #Find ROIs, square sub-grid of the nematic field with POIs at their centre
    ROIs = ROICropper(grid_t,POI_indices,offset)
    ROIs[ROIs<-np.pi/2] += np.pi
    ROIs[ROIs>np.pi/2] -= np.pi
    
    return POIs,ROIs

def GridDirectors(cell_data,xg,yg,offset):
    #Interpolate cell data to fine grid, interpolating trig functions to ensure 
    #smooth variation
    grid_c2t = griddata(cell_data[:,:2],np.cos(2*cell_data[:,2]),(xg,yg), method='linear', fill_value=0)
    grid_s2t = griddata(cell_data[:,:2],np.sin(2*cell_data[:,2]),(xg,yg), method='linear', fill_value=0)
    
    ### Smooth data ###
    grid_c2t_smooth = np.zeros((np.size(grid_c2t,0),np.size(grid_c2t,1)))
    grid_s2t_smooth = np.zeros((np.size(grid_c2t,0),np.size(grid_c2t,1)))
    for i in range(np.size(grid_c2t,0)):
        i_l = i - offset
        i_r = i + offset
        if i_l < 0:
            i_l = 0
        if i_r >= np.size(grid_c2t,0):
            i_r = np.size(grid_c2t,0) - 1
        
        for j in range(np.size(grid_c2t,1)):
            j_l = j - offset
            j_r = j + offset
            if j_l < 0:
                j_l = 0
            if j_r >= np.size(grid_c2t,1):
                j_r = np.size(grid_c2t,1) - 1
                
            grid_c2t_smooth[i,j] = np.mean(grid_c2t[i_l:i_r,j_l:j_r])
            grid_s2t_smooth[i,j] = np.mean(grid_s2t[i_l:i_r,j_l:j_r])
    ### ---------- ###
    
    #Ensure angles are properly are pi periodic after inverting trig functions
    grid_t = np.arccos(grid_c2t_smooth)
    grid_t[grid_s2t_smooth<0] = 2*np.pi - grid_t[grid_s2t_smooth<0]
    grid_t *= 0.5
    
    return grid_t

#Find scalar nematic order parameter at each point nematic field
def SFinder(grid_t,offset):
    grid_c2t = np.cos(2*grid_t)
    grid_s2t = np.sin(2*grid_t)

    m = 2*offset + 1 #Size of smoothing window
    #S field domain is 2*offset smaller than nematic field domain due to smoothing
    #window. This stops the model searching for defects too close to the boundary
    #(where things get very noisy
    S = np.zeros((np.size(grid_t,0) - (m-1), np.size(grid_t,1) - (m-1))); #Initialise S field
    for i in range(np.size(S,0)):
        for j in range(np.size(S,1)):
             S[i,j] = np.sqrt(np.nanmean(grid_c2t[i:i+(m-1),j:j+(m-1)])**2 + np.nanmean(grid_s2t[i:i+(m-1),j:j+(m-1)])**2)
    return S

#Find centres of ROIs
def POIFinder(grid_t,S,offset):
    S_th = 0.15     #Set threshold value of S for ROIs
    S_mask = np.zeros((np.size(S,0),np.size(S,1)),dtype=int)
    #create mask and find contiguous regions of low S
    S_mask[S < S_th] = 1
    labelled = skm.label(S_mask)
    rois = skm.regionprops(labelled) 
    
    c = 0
    for region in rois:
        #Find region centroid
        x_c, y_c = region.centroid
        com = np.round(np.array([[x_c,y_c]]))
        com = com.astype(int)
        #Don't take any ROIs at the edge of the domain
        if np.isnan(grid_t[com[0,0]-offset:com[0,0]+offset+1,com[0,1]-offset:com[0,1]+offset+1]).any() == True:
            continue
        elif np.shape(grid_t[com[0,0]-offset:com[0,0]+offset+1,com[0,1]-offset:com[0,1]+offset+1]) != (9,9):
            continue
        
        #Add first centroid to POI list
        if c == 0:
            poi = com
            c = 1
            continue
        #Add other centroids to POI list
        #But don't accept centroids too close to one another
        #(this happens very rarely but including this makes the detection slightly cleaner)
        rel_dist = poi - com
        rel_mag = np.sqrt(rel_dist[:,0]**2 + rel_dist[:,1]**2)
        if rel_mag.any() < 1:
            continue
        else:
            poi = np.vstack((poi,com))
    poi = poi.astype(int) + offset
    return poi

#Crop ROIs around POIs
def ROICropper(grid_t,poi,offset):
    m = 2*offset + 1
    ROIs = np.zeros((np.size(poi,0),m,m))
    i = 0
 
    for point in poi:
        t = grid_t[point[0]-offset:point[0]+offset+1,point[1]-offset:point[1]+offset+1]
        ROIs[i,:,:] = t
        i= i + 1
        
    return ROIs

#Find defect orientation according to
#Vromans and Giomi, Soft Matter, 2016, 12, 6490-6495
def DefectOrientator(ROIs,k,grid_space):
    #k = defect topological charge
    angles = np.zeros((np.size(ROIs,0),1))

    for i in range(np.size(ROIs,0)):       
        ROI = ROIs[i,:,:];    
        x0_inds = ROI[0,:]
        x2_inds = ROI[-1,:]
        y0_inds = ROI[:,0]
        y2_inds = ROI[:,-1]
        
        #Use central difference scheme to estimate gradients in Q across ROI
        dQxxdx = (np.mean(np.cos(2*x2_inds)) - np.mean(np.cos(2*x0_inds)))/(2*grid_space)
        dQxydy = (np.mean(np.sin(2*y2_inds)) - np.mean(np.sin(2*y0_inds)))/(2*grid_space)
        dQxydx = (np.mean(np.sin(2*x2_inds)) - np.mean(np.sin(2*x0_inds)))/(2*grid_space)
        dQxxdy = (np.mean(np.cos(2*y2_inds)) - np.mean(np.cos(2*y0_inds)))/(2*grid_space)

        numer = np.sign(k)*dQxydx-dQxxdy
        denom = dQxxdx+np.sign(k)*dQxydy
        angles[i] = (k/(1-k))*np.arctan2(np.mean(numer),np.mean(denom))

    return angles
