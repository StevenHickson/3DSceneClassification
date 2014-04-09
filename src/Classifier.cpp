#include "Classifier.h"

using namespace std;
//using namespace pcl;
using namespace cv;


inline void Classifier::CalculateSIFTFeatures(Mat img, Mat keypoints, Mat descriptors) {
	Mat gImg, desc;
	cvtColor(img, gImg, CV_BGR2GRAY);
	vector<KeyPoint> kp;
	featureDetector->detect(gImg,kp);
	descriptorExtractor->compute(gImg,kp,desc);
	keypoints.push_back(kp);
	descriptors.push_back(desc);
}

void Classifier::build_vocab() {

}

void Classifier::load_vocab() {

}