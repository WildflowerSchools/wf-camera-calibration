# camera_calibration

Support for calculating intrinsic and extrinsic camera calibration parameters using COLMAP and other tools

## Task list
* Create endpoint for preparing COLMAP inputs
* Create endpoint for processing COLMAP outputs
* Add error checking to COLMAP calibration `bash` script
* Add more options to COLMAP calibration `bash` script
* Integrate Docker-based COLMAP processing
* Move Honeycomb functionality to separate package
* Move Wildflower-specific functionality into separate package
* Eliminate OpenCV dependency once necessary functionality is in `wf-cv-utils`
