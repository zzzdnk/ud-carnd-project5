## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./report/carnotcar.jpg
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./report/channels.jpg
[image9]: ./report/channels2.jpg
[image10]: ./report/car_hog.jpg
[image11]: ./report/notcar_hog.jpg
[image12]: ./report/windows.jpg
[image13]: ./report/boxes.jpg
[image14]: ./report/heat.jpg
[image15]: ./report/final.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points


##### Writeup / README
You're reading it!

#### Histogram of Oriented Gradients (HOG)

##### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `get_hog_features` contained in the `functions`module.

I started by reading in all the `vehicle` and `non-vehicle` images from the dataset.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces (`['RGB', 'HSV', 'LUV', 'YUV', YCrCb']`) and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I chose random images from each of the two classes and displayed them to get a feel for what the color transform and the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space for the car image

![alt text][image8]

and for the not-car image

![alt text][image9]

As we can see the Y channel contains most relevant information regarding the image. Therefore the method Histogram of Oriented Gradients was applied only to the Y channel of images converted into the YCrCb color space.

Below we can see the HOG features of the car image with parameters `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image10]

and HOG features of the not-car image using the same parameters

![alt text][image11]

##### 2. Explain how you settled on your final choice of HOG parameters.
Since there are a lot of different combinations for these parameters it is hard to choose parameter settings that are optimal to all images in the dataset without risking overfitting to the dataset. I tried various combinations of parameters, trained a classifier using these parameters and compared the performance on the test set. In the end, I settled on my final choice of HOG parameters after viewing the performance of the pipeline on the output video.

Here are the parameters used for the final model
```
color_space = 'YCrCb' # Color space
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0
```


##### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the function `train` located in the file  `project.py`. I trained a linear SVM (Support Vector Machine) classifier using its default parameters. Before training the classifier I split the dataset into randomized training and test set. Training set contained 80% of all datapoints with 20% left for final testing. Each feature vector is obtained in function `single_img_features` by combining spatial features, histogram features and HOG features of a given image.
```
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)
# Use a linear SVM classifier
svc = LinearSVC()
svc.fit(X_train, y_train)
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
```
The final classifier achieved 95.77% accuracy on the test set and was used in the vehicle detection pipeline.

#### Sliding Window Search

##### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
The code for this step is contained in the function `slide_window` located in the file  `functions.py`. I have experimented with different sizes of the windows and different fractions of overlapping. These settings play a great role for the performance of the pipeline. A large window can contain a lot of noise whereas a small window need not be large enough to contain any car.

Further, these settings effect the time required for processing a single frame. The smaller the window size and the larger the overlap, the more time it takes to process the image. Therefore, I had to find a compromise between speed and accuracy. To reduce time requirements further I limited the area to be searched to lower right corner of each image.

Here is an example of an image with sliding windows `128x128` and overlap `0.75`.

![alt text][image12]
Using these values there are 20 windows that need to be searched. If we increase the overlap to `0.85` the window count goes up to 312, and if we further decrease the window size, for example, to `96x96`, we find that there are 700 windows that need to be searched.

Therefore I decided to use sliding windows size `128x128` and overlap `0.85` in the final pipeline. I used these values to get a reasonable speed, it would be better to use smaller window size, especially towards the center of the screen.


##### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched using YCrCb 0-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image13]

I optimized the performance of the classifier mainly by changing the overlapping factor and the window size. The higher the overlap the better chance there is for capturing the vehicles in windows and so we get better detection but at the same time,  time requirements on processing a single image increase significantly.

#### Video Implementation

##### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
The code for this step is contained in  functions `process_image` and `heat_frames` located in the file  `project.py`.  I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here is an example of bounding boxes for an image together with their heatmap

![alt text][image14]

and the final result after applying thresholding and  `scipy.ndimage.measurements.label()` to the heatmaps.

![alt text][image15]

To filter for false positives in the video I combined heatmaps from consecutive video frames and thresholded the resulting heatmap. I tried different number of frames to smoothe the boxes. More frames result into a smoother image but at the cost of processing time. In the end, I obtained the result by combing heatmaps of 3 frames. The threshold parameter can be further tuned to limit false positive detections. For example, if we increase it to 2 then if a false positive appears in a single frame it will not appear in the rendered frame.

#### Discussion

##### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue I faced during the implementation is an appropriate selection of parameters used for features selection and sliding windows size. Also, I was lacking raw processor power to render the video in reasonable time. In the final implementation each frame takes roughly 1.3s to process. If I wanted to finetune the pipeline by decreasing the sliding windows and further filtering over heatmaps, the time could easily increase to 10 seconds or more for a single frame. The speed could be further improved by limiting the search to an area that is likely to contain vehicles based on previous detection.

My current pipeline is likely to fail under different light settings and road conditions. The pipeline could be improved by making sure no vehicle suddenly disappears, and no physically unrealistic boxes are found in a video frame. Moreover, a prediction of likely position could be done by computing positions of vehicles in previous frames.

In the next project I would like to investigate applications of convolutional neural networks for vehicle detection. Especially I would like to compare performancce of recent architectures like YOLO ([You Only Look Once](https://pjreddie.com/darknet/yolo/)) to my pipeline.
