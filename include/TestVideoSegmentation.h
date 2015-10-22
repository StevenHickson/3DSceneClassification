#include <iostream>

#include "Microsoft_grabber.h"
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/opencv.hpp>
#include "Classifier.h"
#include "Edges.h"
#include "OpticalFlow.h"
#include "GraphSegmentation.h"

#include <Shlwapi.h>
#include <string.h>

#define NUM_LABELS 894

inline void EstimateNormals(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud, pcl::PointCloud<pcl::PointNormal>::Ptr &normals);
void BuildRFClassifier(std::string direc);
void BuildNYUDataset(std::string direc, bool matlab = false);
void BuildNYUDatasetForCaffe(std::string direc);
void TestRFClassifier(std::string direc);