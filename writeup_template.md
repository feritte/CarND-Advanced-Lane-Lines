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

[image1]: ./examples/undistorted.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/ResultofThresholding.png "Binary Example"
[image4]: ./examples/ResultofWrapped.png "Warp Example"
[image5]: ./examples/PolyNomFit.png "Fit Visual"
[image6]: ./examples/LaneProjected.png "Output"
[image7]: ./examples/codeUndist.PNG "Code for undistortion"
[image8]: ./examples/codeThreshold.PNG "Code for thresholding"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or through the first cells of the file called `p4solution.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The code partion for that is given below

![alt text][image7]

![alt text][image8]

Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `apply_birds_eye(()`, which appears in the file `p4solution.ipynb`.  The `apply_birds_eye(()` function takes as inputs an image (`img`).

```python
def apply_birds_eye(img, should_display=True):
    img_shape = (img.shape[1], img.shape[0])
    
    src = np.float32(SRC_PTS)
    dst = np.float32(DST_PTS)
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_shape)
    
    if should_display is True:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Undistorted Image', fontsize=20)
        ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        ax2.set_title('Warped Image', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    return warped, M, Minv
    ```
    
Source (`src`) and destination (`dst`) points are given seperately.  I chose the hardcode the source and destination points in the following manner:

```python
SRC_PTS = [[490, 482],[810, 482],[1250, 720], [40, 720]]
tl = SRC_PTS[0]
tr = SRC_PTS[1]
br = SRC_PTS[2]
bl = SRC_PTS[3]
# compute the width of the new image, which will be the
# maximum distance between bottom-right and bottom-left
# x-coordiates or the top-right and top-left x-coordinates
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))
 
# compute the height of the new image, which will be the
# maximum distance between the top-right and bottom-right
# y-coordinates or the top-left and bottom-left y-coordinates
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))
print(maxWidth)
print(maxHeight)
DST_PTS_test = np.array([
[0, 0],
[maxWidth - 1, 0],
[maxWidth - 1, maxHeight - 1],
[0, maxHeight - 1]], dtype = "float32")
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 490, 482      | 0, 0        | 
| 810, 482      | 1280, 0      |
| 1250, 720     | 1250, 720      |
|  40, 720      | 40, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

where the lane lines stand out clearly. However, you still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

I first take a histogram along all the columns in the lower half of the image like this:
```python
def apply_binary_thresholds(img, thresholds={  \
      's': {'min': 180, 'max': 255}, \
      'l': {'min': 255, 'max': 255},   \
      'b': {'min': 155, 'max': 200}  \
    } , should_display=True): 
    
    S = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]  
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,0]
    B = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:,:,2] 

    s_bin = np.zeros_like(S)
    s_bin[(S >= thresholds['s']['min']) & (S <= thresholds['s']['max'])] = 1
    b_bin = np.zeros_like(B)
    b_bin[(B >= thresholds['b']['min']) & (B <= thresholds['b']['max'])] = 1
    l_bin = np.zeros_like(L)
    l_bin[(L >= thresholds['l']['min']) & (L <= thresholds['l']['max'])] = 1
    
    full_bin = np.zeros_like(s_bin)
    full_bin[(l_bin == 1) | (b_bin == 1) | (s_bin == 1)] = 1

    if should_display is True:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
        f.tight_layout()
        ax1.set_title('original image', fontsize=16)
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8'))
        ax2.set_title('all thresholds', fontsize=16)
        ax2.imshow(full_bin, cmap='gray')
        
    return full_bin
```

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
* Peaks are identified in a histogram of the image to determine location of lane lines.
* All non zero pixels around histogram peaks using the numpy function numpy.nonzero() is found.
* A two degree polynomial to each lane using the numpy function numpy.polyfit() is fitted.
* The average of the x intercepts from each of the two polynomials position = (rightx_int+leftx_int)/2 is calculated
* The distance from center by taking the absolute value of the vehicle position minus the halfway point along the horizontal axis distance_from_center = abs(image_width/2 - position) is calculated.
*  if The horizontal position of the car was greater than image_width/2 than the car was considered to be left of center, otherwise right of center.
* The distance from center was converted from pixels to meters by multiplying the number of pixels by 3.7/700.
* The following code is used to calculate the radius of curvature for each lane line in meters:
```python
def annotate(img, curvature, pos, curve_min):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'curvature radius = %d(m)' % (curvature / 128 * 3.7), (50, 50), font, 1, (255, 255, 255), 2)
    cv2.putText(img, '%.2fm %s of center' % (np.abs(pos / 12800 * 3.7), "left"), (50, 100), font, 1,
                (255, 255, 255), 2)
    cv2.putText(img, 'curvature minimum radius = %d(m)' % (curve_min / 128 * 3.7), (50, 150), font, 1, (255, 255, 255), 2)

def pos_cen(y, left_poly, right_poly):
    return (1.5 * polynomial_lines(y, left_poly)
              - polynomial_lines(y, right_poly)) / 2

lc_radius = np.absolute(((1 + (2 * lcs[0] * 500 + lcs[1])**2) ** 1.5) \
                /(2 * lcs[0]))
rc_radius = np.absolute(((1 + (2 * rcs[0] * 500 + rcs[1]) ** 2) ** 1.5) \
                 /(2 * rcs[0]))

ll_img = cv2.add( \
    cv2.warpPerspective( \
        painted_b_eye, Minv, (shape[1], shape[0]), flags=cv2.INTER_LINEAR \
    ), undistorted \
) 
plt.imshow(ll_img)
annotate(ll_img, curvature=(lc_radius + rc_radius) / 2, 
                     pos=pos_cen(719, lcs, rcs), 
                     curve_min=min(lc_radius, rc_radius))
plt.imshow(ll_img)

```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines of my code in the function `annotate()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
