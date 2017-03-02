# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 15:34:50 2017

@author: Zdenek
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from functions import *
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

#Find the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    """
    Find the list of windows to be searched
    """
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [350, 550] # Min and max in y to search in slide_window()
    
def train():
    """
    Train the SVM classifier on the dataset
    """    
    
    cars = glob.glob('./carornot_big/vehicles/*/*.png')
    notcars = glob.glob('./carornot_big/non-vehicles/*/*.png')
            
            
    # Reduce the sample size because
    sample_size = -1
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]
    
    time1 = time.time()
    
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    
    print('Features extracted in {:.3f} s'.format(time.time() - time1))
    
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    
    # Split up data into randomized training and test sets
    rand_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

    # Save the classifier 
    import pickle
    d = {}
    d['svc'] = svc
    d['scaler'] = X_scaler
    d['orient'] = orient
    d['hist_bins'] = hist_bins
    d['spatial_size'] = spatial_size
    d['cell_per_block'] = cell_per_block
    d['pix_per_cell'] = pix_per_cell
    with open('svc_pickle.p', 'wb') as f:
        pickle.dump(d, f)

#%%
import pickle
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
      
        
#%%
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
import pandas as pd
from scipy.ndimage.measurements import label


def process_image(draw_image, debug=False):
    image = np.copy(draw_image)
    image = draw_image.astype(np.float32)/255

    windows = slide_window(image, x_start_stop=[650, None], y_start_stop=y_start_stop, 
                    xy_window=(128, 128), xy_overlap=(0.85, 0.85))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

    if debug:
        boxmap = np.copy(draw_image)
        boxmap = draw_boxes(boxmap, hot_windows, color=(0, 0, 255), thick=6)
        plt.imshow(boxmap), plt.show()
        mpimg.imsave('box6.jpg', boxmap)
    
    heatmap = np.zeros((720, 1280))
    
    heatmap = add_heat(heatmap, hot_windows)
    hheat = np.max(heatmap)
    if debug: 
        print('hightest heat: {}'.format(hheat))
        plt.figure(figsize=(15,10))
        plt.subplot(121)
        plt.imshow(boxmap)
        plt.subplot(122), plt.imshow(heatmap), plt.show()
    heatmap = apply_threshold(heatmap, 1)
    
    labels = label(heatmap)
    if debug: print(len(labels))
    if debug: print(labels[1], 'cars found')
    if labels[1] > 0:
     #   plt.imshow(labels[0]), plt.show()
        #result = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
        result = draw_labeled_bboxes(draw_image,labels)                    

    else:
        result = draw_image
    return result
    
def process_frames(get_frame, t):
  
    image = get_frame(t)
    result = process_image(image)
    #cv2.putText(result,"Frame: {}".format(np.int(30*t)), (100,200), cv2.FONT_HERSHEY_DUPLEX, 1, 100,3)

    return result

def batch_process_frames(get_frame, t):
  
    frames = []
    for i in range(3):
        frames.append(get_frame(t + i/25))
    #image = get_frame(t)
    result = heat_frames(frames)
    #cv2.putText(result,"Frame: {}".format(np.int(30*t)), (100,200), cv2.FONT_HERSHEY_DUPLEX, 1, 100,3)

    return result
    
image = mpimg.imread('./test_images/test6.jpg')
plt.imshow(process_image(image, True)), plt.show()
#plt.imshow(draw_labeled_bboxes(np.copy(image), labels))



#%%
def heat_frames(frames, debug=False):

    heatmap = np.zeros((720, 1280))    
    
    for image in frames:
        draw_image = np.copy(image)
        image = draw_image.astype(np.float32)/255
        windows = slide_window(image, x_start_stop=[650, None], y_start_stop=y_start_stop, 
                        xy_window=(128, 128), xy_overlap=(0.85, 0.85))
    
        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
    

        heatmap = add_heat(heatmap, hot_windows)

    if debug:
        print('hightest heat: {}'.format(np.max(heatmap)))
        plt.imshow(heatmap), plt.show()
    heatmap = apply_threshold(heatmap, 2)
    labels = label(heatmap)
    if debug: print(labels[1], 'cars found')
    if labels[1] > 0:
        if debug: plt.imshow(labels[0]), plt.show()
     #   result = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
        result = draw_labeled_bboxes(draw_image,labels)                    

    else:
        result = draw_image
    return result



video = 'project_video'

output = video + '_out.mp4'
clip = VideoFileClip('./' + video + '.mp4')
#subclip = clip.subclip(23, 30)
#○subclip = clip.subclip(25, 30)
subclip = clip
#out_clip = subclip.fl(process_frames) 
out_clip = subclip.fl(batch_process_frames) 
out_clip.write_videofile(output, audio=False)

