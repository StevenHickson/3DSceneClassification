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

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>

#include "OpticalFlow.h"
#include "GraphSegmentation.h"

#include <Shlwapi.h>
#include <string.h>

#define NUM_CLUSTERS 500;

cv::Mat imread_depth(const char* fname, bool binary);
void CreatePointCloudFromRegisteredNYUData(const cv::Mat &img, const cv::Mat &depth, PointCloudBgr *cloud);
void LoadData(std::string direc, int i, cv::Mat &img, cv::Mat &depth, cv::Mat &label);

class Classifier {
public:
	int categories; //number of categories
	int clusters; //number of clusters for SURF features to build vocabulary
	std::string direc; //directory of NYU data
	cv::Mat vocab; //vocabulary
	cv::Ptr<cv::FeatureDetector> featureDetector;
	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
	cv::Ptr<cv::BOWKMeansTrainer> bowtrainer;
	cv::Ptr<cv::BOWImgDescriptorExtractor> bowDescriptorExtractor;
	cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;

	std::deque<int> testingInds, trainingInds;
	std::vector<int> classMap;

	Classifier(std::string direc_) {
		direc = direc_;
		clusters = NUM_CLUSTERS;
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
	void CalculateSIFTFeatures(cv::Mat &img, cv::Mat &mask, cv::Mat &descriptors);
	void CalculateBOWFeatures(cv::Mat &img, cv::Mat &mask, cv::Mat &descriptors);
	void LoadTestingInd();
	void LoadTrainingInd();
	void LoadClass4Map();

};

#endif