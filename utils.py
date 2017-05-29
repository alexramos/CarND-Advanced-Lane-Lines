import numpy as np
import cv2

# Constants for warping/unwarping functions
IMSHAPE = (720, 1280) # hardcoded
SRC_POINTS = np.float32([
    (0 + 185, IMSHAPE[0]),
    (IMSHAPE[1]/2 - 55, 455),
    (IMSHAPE[1]/2 + 55, 455),
    (IMSHAPE[1] - 145, IMSHAPE[0])
])

DEST_POINTS = np.float32([
    (0 + 325, IMSHAPE[0]),
    (0 + 325, 0),
    (IMSHAPE[1] - 325, 0),
    (IMSHAPE[1] - 325, IMSHAPE[0])])

def warper(img):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(SRC_POINTS, DEST_POINTS)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def unwarper(img):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(DEST_POINTS, SRC_POINTS)
    dewarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return dewarped

def get_binary_img(image, ksize=15):
    def abs_sobel_thresh(l_channel, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient    
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        elif orient == 'y':
            sobel = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)

        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)

        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # 5) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return grad_binary

    def mag_thresh(l_channel, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)

        # 3) Calculate the magnitude 
        abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # 5) Create a binary mask where mag thresholds are met
        mag_binary = np.zeros_like(scaled_sobel)
        mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return mag_binary

    def dir_threshold(l_channel, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        grad_dir = np.arctan2(abs_sobely, abs_sobelx)

        # 5) Create a binary mask where direction thresholds are met
        dir_binary = np.zeros_like(grad_dir)
        dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return dir_binary

    def color_threshold(s_channel, thresh=(0, 255)):  
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

        return s_binary

    
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(l_channel, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(l_channel, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(l_channel, sobel_kernel=ksize, thresh=(0.7, 1.3))
    col_binary = color_threshold(s_channel, thresh=(170, 255))

    combined_grad = np.zeros_like(dir_binary)
    combined_grad[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined = np.zeros_like(combined_grad)
    combined[((combined_grad == 1) | (col_binary == 1))] = 1
    
    return combined

def fit_lane_lines(binary_warped, margin=75, minpix=50):
    # margin: the width of the windows +/- margin
    # minpix: set minimum number of pixels found to recenter window
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2 ):,:], axis=0)

    # Trim noisy ends from influencing 
    histogram[:80]= 0
    histogram[-80:] = 0
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
     
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height;
        win_y_high = binary_warped.shape[0] - window*window_height;
        win_xleft_low = leftx_current - margin;
        win_xleft_high = leftx_current + margin;
        win_xright_low = rightx_current - margin;
        win_xright_high = rightx_current + margin;

        # Draw the windows on the visualization image
        _ = cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2);
        _ = cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2);

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
            (nonzerox >= win_xleft_low) & \
            (nonzerox < win_xleft_high)
        ).nonzero()[0];
        good_right_inds = (
            (nonzeroy >= win_y_low) & \
            (nonzeroy < win_y_high) & \
            (nonzerox >= win_xright_low) & \
            (nonzerox < win_xright_high)
        ).nonzero()[0];

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds);
        right_lane_inds.append(good_right_inds);

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]));
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]));
            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return left_fit, right_fit, out_img

def fit_lane_lines_from_previous_fit(binary_warped, left_fit, right_fit, margin=75):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = (
        (nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & \
        (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))
    ) 
    right_lane_inds = (
        (nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & \
        (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin))
    )
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    new_left_fit = np.polyfit(lefty, leftx, 2)
    new_right_fit = np.polyfit(righty, rightx, 2)
    
    return new_left_fit, new_right_fit

def draw_detected_lane(undist, binary_warped, ploty, left_fitx, right_fitx):  
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0));
    newwarp = unwarper(color_warp)

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result
