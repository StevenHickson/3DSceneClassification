/*
Copyright (C) 2014 Steven Hickson

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA

*/

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>

class Classifier {
public:
	int categories; //number of categories
	int clusters; //number of clusters for SURF features to build vocabulary
	cv::Mat vocab; //vocabulary
	cv::Ptr<cv::FeatureDetector> featureDetector;
	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
	cv::Ptr<cv::BOWKMeansTrainer> bowtrainer;
	cv::Ptr<cv::BOWImgDescriptorExtractor> bowDescriptorExtractor;
	cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;

	Classifier() {
		clusters = 1000;
		categories = 4;
		featureDetector = (new cv::SurfFeatureDetector());
		descriptorExtractor = (new cv::SurfDescriptorExtractor());
		bowtrainer = (new cv::BOWKMeansTrainer(clusters));
		descriptorMatcher = (new cv::FlannBasedMatcher());
		//descriptorMatcher = (new BFMatcher());
		bowDescriptorExtractor = (new cv::BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher));
	};

	void build_vocab(); //function to build the BOW vocabulary
	void load_vocab(); //function to load the BOW vocabulary and classifiers
	inline void CalculateSIFTFeatures(cv::Mat img, cv::Mat keypoints, cv::Mat descriptors);

};

#endif