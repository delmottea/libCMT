libCMT
======

c++ port of the python code from https://github.com/gnebehay/CMT

based on paper Consensus-based Matching and Tracking of Keypoints for Object Tracking, Nebehay, Georg and Pflugfelder, Roman, 2014

Notes
=====
If the initialise function is slow (seems to be the case in android), try to move the 2 lines :

detector = cv::Algorithm::create<cv::FeatureDetector>(detectorType.c_str());
descriptorExtractor = cv::Algorithm::create<cv::DescriptorExtractor>(descriptorType.c_str());

to the constructor.
