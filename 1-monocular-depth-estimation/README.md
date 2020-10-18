# MONOCULAR DEPTH ESTIMATION AND SEGMENTATION SIMULTANEOUSLY

## Project Description

In this project custom dataset of around half million images are created and various CNN Network are build that can do monocular depth estimation and foreground-background separation simultaneously. 

> Applications: 
- For autonomous bots like  cleaning bot we can use this model for real-time tracking of changes in the environment by using foreground-background seperation, and could able to find the distance with depth estimation.
- Automatically opening of door only when person is near to the door. 

The MONOCULAR DEPTH ESTIMATION AND SEGMENTATION network should take two images as input.

1. Background image(bg).
2. Foreground overlayed background image(fg_bg).

And it should give two images as output.

* Depth estimation image.
* Mask for the foreground in background overlayed image.

## Overall work Summary:

1. Custom dataset of size half million images are created. [(Link)](dataset_creation)

2. Building a model for mask and depth map prediction for custom dataset where foreground are overlay over background and it equivalent ground truth mask and depth are given as target images.
 * Custom Dataset and data loaded functions
 * Data split: 70:30(train:test)
 * Image augmentation techniques
 * Experiment with different Loss functions
 * Technique for accuracy calculation
 * Experiment with various model architecture
 * Strategy for beating 12 hours colab limitation run

**For complete details on how solution is developed, refer [(Link)](solutions)**

 


