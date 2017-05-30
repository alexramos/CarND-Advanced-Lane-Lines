# Writeup 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_chessboard.png "Undistorted Chessboard"
[image2]: ./output_images/undistorted_test_img.png "Undistored Test"
[image3]: ./output_images/binary_test_img.png "Binary Test"
[image4]: ./output_images/transformed_test_img.png "Warp Test"
[image7]: ./output_images/transformed_binary_img.png "Warp Binary"
[image5]: ./output_images/lane_pixels_binary_img.png "Fit Visual"
[image8]: ./output_images/lane_pixels_2_binary_img.png "Fit 2 Visual"
[image6]: ./output_images/final_output_img.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "lan\e_finding\_dev.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Below is an example of distortion correction applied to one of the test images.  Note the difference in the car's body in the lower left and right corners of the undistored image.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (detailed in the 6th and 7th code cells in "lane\_finding\_dev.ipynb").

Valid pixels to be used for lane finding met the following criteria:

- HLS saturation value met threshold constraints

OR

- HLS lightness value met combinbed gradient constraints
	- Valid gradient in both x AND y direction

	OR
	
	- Valid gradient magnitude AND direction

Below is an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in the 8th code cell of "lane\_finding\_dev.ipynb").  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  The source and destination points are hardcoded to 1280x720 images in the following manner:

```python
imshape = (720, 1280)
src_points = np.float32([
    (0 + 185, imshape[0]),
    (imshape[1]/2 - 55, 455),
    (imshape[1]/2 + 55, 455),
    (imshape[1] - 145, imshape[0])
])

dest_points = np.float32([
    (0 + 325, imshape[0]),
    (0 + 325, 0),
    (imshape[1] - 325, 0),
    (imshape[1] - 325, imshape[0])])

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 185, 720      | 325, 720        | 
| 585, 455      | 325, 0      |
| 695, 455     | 955, 0      |
| 1135, 720      | 955, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Below are perspective transforms on a rgb camera image and on a "binary" image.

![alt text][image4]
![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Identification of lane-line pixels is described in code cells 10-15 in "lane\_finding\_dev.ipynb".  Using a binary perspective-transformed image as input, I perform the following:

1. Compute a histogram of the bottom half of the image to find positions with most pixels.
2. Find the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines.
3. Scan the image upwards starting from the bottom in two series of nine rectangular windows.  Each window defines the pixels that will be to compute the linear fit for each lane line.  During scanning, window positions are adjusted to the mean pixel position within the window.
4. Compute a 2nd order polynomial fit from the pixel positions in each series of  windows.

![alt text][image5]

If lane lines have already been computed from prior images, I compute a new fit by using pixels located within a margin around the previous fit (see example below):

![alt text][image8]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Calculation of lane curvature radius and vehicle position is described in code cells 16-17 in "lane\_finding\_dev.ipynb".

To determine the radius of curvature of the lane, I first convert my two lane curves from pixel coordinates to real-world curves in meters.  I then compute radii from two new 2nd order polynomials fitted to the real-world curves.  Lastly, I average the two lane-line curvature radii to compute a single curvature radius for the lane.

For vehicle position with respect to the center of lane, we assume that the image's center is also the center of the car.  To determine the center of the lane we use the midpoint between the base positions for each lane line.  Thus, the vehicle's position from the center of the lane is the difference between the image center and the lane center.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell 18 in "lane\_finding\_dev.ipynb".  The two lane-line curves are used to draw an image of a polygon denoting the lane.  This image is then unwarped (perspective transformed) and overlaid on the original undistorted image.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The pipeline for processing video input is described in "lane\_finding\_pipeline.ipynb".

**Pipeline steps:**

1. Undistort image with saved camera calibrations
2. "Binarize" image using color and gradient
3. Fit lane lines using A. (scanning) or B. (look-ahead filter) approaches
    1. Scan image for lane pixels using windowed approach
    2. Search for lane pixels around existing fit
4. Perform sanity check on lane-line fits
    1. Ensure lines are reasonable distance apart
    2. Ensure lines are roughly parallel
5. Draw lane fit onto image


- When fitting lane lines, approach B. (look-ahead filter) is used if previous good fits exists in memory.
- For drawing the detected lane, the output is smoothed over the previous 30 frames.
- If sanity check fails for a fit, the last good fit is used.  If sanity check fails 15 consecutive frames, the pipeline falls back to searching for a fit using approach A. (scanning).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took is largely based off the examples provided in the this section's lessons.  I first use calibration chessboard images to correct for image distortion introduced by the camera's lens.  I then identify line pixels in the driving image using color and gradient thresholding.  Theses pixels are perspective-transformed to a 'birds-eye' view and used to determine a 2nd order polynomial fit for each lane line.  Finally, these fits are used to define the lane's position and overlaid over the original image.

My implementation works very well for the target project video but performs very poorly with the challenge videos.  Specifically, my algorithm fails when it has to deal with muliple vertical lines in a image (as in "challenge\_video.mp4") or very sharp turns (as in "harder\_challenge\_video.mp4").

If I were going to persue this project further, I would make the pipeline more robust by tuning the pipeline to handle the challenge cases as well.  This could entail:

- Tuning image binarization to better handle shadows
- Tune window-scanning to correctly track sharp curves
- Try convolution approach for finding best window centers (as described in lesson 34.)
- Use a deep-learning approach to find the lane.


---