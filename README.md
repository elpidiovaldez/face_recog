This is a face recognition package for ROS2 (tested on Humble).

It uses Facenet to do the work, and is trained on your own data.  You should supply 
around 20 photos of each person to train it.

The input is a ROS2 topic containing video (produced by any ROS2 camera node).

The output is a ROS2 topic containing the bounding boxes of detected faces and the 
identity of the person in the bounding box.

See BUILD_INSTRUCTIONS and USAGE_INSTRUCTIONS for more info.
