Authors: Weixin Liu, Changhao Yang, Hong Chen

This repository is for 18-645 Project: Gaussian Blur Optimization

This project is built by using xcode IDE from apple inc..

All source codes are located in 18645_Project folder. A README providing a high level overview of the code is also located in the same folder.

Main.cpp contains the main SIFT implementation which is from OpenCV. The kernel
for convolution with column filter is defined in vert_conv.cpp. The kernel for
matrix-matrix substraction is defined in GaussianBlur_modified.cpp.

In the main.cpp, for gaussian blur, we are replacing OpenCV's implementation 
GuassianBlur in line 210 with our version GaussianBlur_modified in line 211.
For matrix-matrix substraction, we are replacing OpenCV's implementation
subtract in line 242 with our version substraction_kernel in line 242.
