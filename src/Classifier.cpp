#include "Classifier.h"

using namespace std;
using namespace pcl;
using namespace cv;

Mat imread_depth(const char* fname, bool binary) {
	char* ext = PathFindExtension(fname);
	const char char_dep[] = ".dep";
	const char char_png[] = ".png";
	Mat out;
	if(_strnicmp(ext,char_dep,strlen(char_dep))==0) {
		FILE *fp;
		if(binary)
			fp = fopen(fname,"rb");
		else
			fp = fopen(fname,"r");
		int width = 640, height = 480; //If messed up, just assume
		if(binary) {
			fread(&width,sizeof(int),1,fp);
			fread(&height,sizeof(int),1,fp);
			out = Mat(height,width,CV_32S);
			int *p = (int*)out.data;
			fread(p,sizeof(int),width*height,fp);
		} else {
			//fscanf(fp,"%i,%i,",&width,&height);
			out = Mat(height,width,CV_32S);
			int *p = (int*)out.data, *end = ((int*)out.data) + out.rows*out.cols;
			while(p != end) {
				fscanf(fp,"%i",p);
				p++;
			}
		}
		fclose(fp);
	} else if(_strnicmp(ext,char_png,strlen(char_png))==0) {
		out = cvLoadImage(fname,CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
		out.convertTo(out, CV_32S);
		int* pi = (int*)out.data;
		for (int y=0; y < out.rows; y++) {
			for (int x=0; x < out.cols; x++) {
				*pi = Round(*pi * 0.2f);
				pi++;
			}
		}
	} else {
		throw exception("Filetype not supported");
	}
	return out;
}

Mat imread_float(const char* fname, bool binary) {
	char* ext = PathFindExtension(fname);
	const char char_dep[] = ".flt";
	Mat out;
	if(_strnicmp(ext,char_dep,strlen(char_dep))==0) {
		FILE *fp;
		if(binary)
			fp = fopen(fname,"rb");
		else
			fp = fopen(fname,"r");
		int width = 640, height = 480; //If messed up, just assume
		if(binary) {
			fread(&width,sizeof(int),1,fp);
			fread(&height,sizeof(int),1,fp);
			out = Mat(height,width,CV_32F);
			float *p = (float*)out.data;
			fread(p,sizeof(float),width*height,fp);
		} else {
			//fscanf(fp,"%i,%i,",&width,&height);
			out = Mat(height,width,CV_32F);
			float *p = (float*)out.data, *end = ((float*)out.data) + out.rows*out.cols;
			while(p != end) {
				fscanf(fp,"%f",p);
				p++;
			}
		}
		fclose(fp);
	} else {
		throw exception("Filetype not supported");
	}
	return out;
}

void imwrite_depth(const char* fname, Mat &img, bool binary) {
	char* ext = PathFindExtension(fname);
	const char char_dep[] = ".dep";
	Mat out;
	if(_strnicmp(ext,char_dep,strlen(char_dep))==0) {
		FILE *fp;
		if(binary)
			fp = fopen(fname,"wb");
		else
			fp = fopen(fname,"w");
		int width = img.cols, height = img.rows; //If messed up, just assume
		if(binary) {
			fwrite(&width,sizeof(int),1,fp);
			fwrite(&height,sizeof(int),1,fp);
			int *p = (int*)img.data;
			fwrite(p,sizeof(int),width*height,fp);
		} else {
			//fscanf(fp,"%i,%i,",&width,&height);
			int *p = (int*)img.data, *end = ((int*)img.data) + width*height;
			while(p != end) {
				fscanf(fp,"%f",p);
				p++;
			}
		}
		fclose(fp);
	} else {
		throw exception("Filetype not supported");
	}
}

void imwrite_float(const char* fname, Mat &img, bool binary) {
	char* ext = PathFindExtension(fname);
	const char char_dep[] = ".flt";
	Mat out;
	if(_strnicmp(ext,char_dep,strlen(char_dep))==0) {
		FILE *fp;
		if(binary)
			fp = fopen(fname,"wb");
		else
			fp = fopen(fname,"w");
		int width = img.cols, height = img.rows; //If messed up, just assume
		if(binary) {
			fwrite(&width,sizeof(int),1,fp);
			fwrite(&height,sizeof(int),1,fp);
			float *p = (float*)img.data;
			fwrite(p,sizeof(float),width*height,fp);
		} else {
			//fscanf(fp,"%i,%i,",&width,&height);
			float *p = (float*)img.data, *end = ((float*)img.data) + width*height;
			while(p != end) {
				fscanf(fp,"%f",p);
				p++;
			}
		}
		fclose(fp);
	} else {
		throw exception("Filetype not supported");
	}
}

void Classifier::CalculateSIFTFeatures(Mat &img, Mat &mask, Mat &descriptors) {
	Mat gImg, desc;
	cvtColor(img, gImg, CV_BGR2GRAY);
	vector<KeyPoint> kp;
	featureDetector->detect(gImg,kp,mask);
	descriptorExtractor->compute(gImg,kp,desc);
	//keypoints.push_back(kp);
	descriptors.push_back(desc);
}

void Classifier::CalculateBOWFeatures(Mat &img, Mat &mask, Mat &descriptors) {
	Mat gImg, desc;
	cvtColor(img, gImg, CV_BGR2GRAY);
	vector<KeyPoint> kp;
	featureDetector->detect(gImg,kp,mask);
	bowDescriptorExtractor->compute(gImg,kp,desc);
	//keypoints.push_back(kp);
	descriptors.push_back(desc);
}

void Classifier::build_vocab() {
	cout << "Building vocabulary" << endl;
	// Mat to hold SURF descriptors for all templates
	// For each template, extract SURF descriptors and pool them into vocab_descriptors
	cout << "Building SURF Descriptors..." << endl;
	Mat img, descriptors;

	LoadTrainingInd();
	for(int i = 1; i < 1450; i++) {
		if(i == trainingInds.front()) {
			trainingInds.pop_front();
			stringstream num;
			num << i;
			img = imread(string(direc + "rgb\\" + num.str() + ".bmp"));
			CalculateSIFTFeatures(img,Mat(),descriptors);
		}
	}

	// Add the descriptors to the BOW trainer to cluster
	cout << "Training BOW..." << endl;
	bowtrainer->add(descriptors);
	// cluster the SURF descriptors
	cout << "Clustering..." << endl;
	vocab = bowtrainer->cluster();
	//vocab.convertTo(vocab,CV_8UC1);
	cout << "Done." << endl;

	// Save the vocabulary
	FileStorage fs("vocab.xml", FileStorage::WRITE);
	fs << "vocabulary" << vocab;
	fs.release();

	cout << "Built vocabulary" << endl;
}

void Classifier::load_vocab() {
	//load the vocabulary
	FileStorage fs("vocab.xml", FileStorage::READ);
	fs["vocabulary"] >> vocab;
	fs.release();

	// Set the vocabulary for the BOW descriptor extractor
	bowDescriptorExtractor->setVocabulary(vocab);
}

void Classifier::LoadTestingInd() {
	FILE *fp = fopen("testing_ind.txt","rb");
	if(fp == NULL)
		throw exception("Couldn't open testing file");
	int length, i;
	fread(&length,sizeof(int),1,fp);
	testingInds.resize(length);
	for(i = 0; i < length; i++) {
		int tmp;
		fread(&tmp,sizeof(int),1,fp);
		testingInds[i] = tmp;
	}
	fclose(fp);
}

void Classifier::LoadTrainingInd() {
	FILE *fp = fopen("training_ind.txt","rb");
	if(fp == NULL)
		throw exception("Couldn't open training file");
	int length, i;
	fread(&length,sizeof(int),1,fp);
	trainingInds.resize(length);
	for(i = 0; i < length; i++) {
		int tmp;
		fread(&tmp,sizeof(int),1,fp);
		trainingInds[i] = tmp;
	}
	fclose(fp);
}

void Classifier::LoadClass4Map() {
	int length, i;
	FILE *fp = fopen("class4map.txt","rb");
	if(fp == NULL)
		throw exception("Couldn't open class mapping file");
	fread(&length,sizeof(int),1,fp);
	length++;
	classMap.resize(length);
	for(i = 1; i < length; i++) {
		int tmp;
		fread(&tmp,sizeof(int),1,fp);
		classMap[i] = tmp;
	}
	fclose(fp);
}

void LoadData(string direc, int i, Mat &img, Mat &depth, Mat &label) {
	stringstream num;
	num << i;
	img = imread(string(direc + "rgb\\" + num.str() + ".bmp"));
	depth = imread_depth(string(direc + "depth\\" + num.str() + ".dep").c_str(),true);
	label = imread_depth(string(direc + "labels\\" + num.str() + ".dep").c_str(),true);
}