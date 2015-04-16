libCMT
======

c++ port of the python code from https://github.com/gnebehay/CMT

based on paper Consensus-based Matching and Tracking of Keypoints for Object Tracking, Nebehay, Georg and Pflugfelder, Roman, 2014

Notes
=====
You need to download the "cokecan" dataset and put it in "sequences/cokecan/" to test the code (the link is given here : http://www.gnebehay.com/cmt/ ). 
You can modify main.cpp to change the dataset.

If the initialise function is slow (seems to be the case in android), try to move the 2 lines :

detector = cv::Algorithm::create<cv::FeatureDetector>(detectorType.c_str());
descriptorExtractor = cv::Algorithm::create<cv::DescriptorExtractor>(descriptorType.c_str());

to the constructor.
