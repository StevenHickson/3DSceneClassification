#include "TestVideoSegmentation.h"
#include "RegionTree.h"

using namespace std;
using namespace pcl;
using namespace cv;

const float parameters[] = { 2.5f,500.0f,500,0.8f,400.0f,400,100,0.25f };
const string mapFile = "class4map.txt";

void CreatePointCloudFromRegisteredNYUData(const Mat &img, const Mat &depth, PointCloudBgr *cloud) {
	//assert(!img.IsNull() && !depth.IsNull());
	//take care of old cloud to prevent memory leak/corruption
	if (cloud != NULL && cloud->size() > 0) {
		cloud->clear();
	}
	cloud->header.frame_id =  "/microsoft_rgb_optical_frame";
	cloud->height = 480;
	cloud->width = 640;
	cloud->is_dense = true;
	cloud->points.resize (cloud->height * cloud->width);
	PointCloud<PointXYZRGBA>::iterator pCloud = cloud->begin();
	Mat_<int>::const_iterator pDepth = depth.begin<int>();
	Mat_<Vec3b>::const_iterator pImg = img.begin<Vec3b>();
	for(int j = 0; j < img.rows; j++) {
		for(int i = 0; i < img.cols; i++) {
			pCloud->z = *pDepth / 1000.0f;
			pCloud->x = float((i - 3.1304475870804731e+02) * pCloud->z / 5.8262448167737955e+02);
			pCloud->y = float((j - 2.3844389626620386e+02) * pCloud->z / 5.8269103270988637e+02);
			pCloud->b = (*pImg)[0];
			pCloud->g = (*pImg)[1];
			pCloud->r = (*pImg)[2];
			pCloud->a = 255;
			pImg++; pDepth++; pCloud++;
		}
	}
	cloud->sensor_origin_.setZero ();
	cloud->sensor_orientation_.w () = 1.0;
	cloud->sensor_orientation_.x () = 0.0;
	cloud->sensor_orientation_.y () = 0.0;
	cloud->sensor_orientation_.z () = 0.0;
}

void CreateLabeledCloudFromNYUPointCloud(const PointCloudBgr &cloud, const Mat &label, PointCloudInt *labelCloud) {
	if (labelCloud != NULL && labelCloud->size() > 0) {
		labelCloud->clear();
	}
	labelCloud->header.frame_id =  cloud.header.frame_id;
	labelCloud->height = cloud.height;
	labelCloud->width = cloud.width;
	labelCloud->is_dense = true;
	labelCloud->points.resize (labelCloud->height * labelCloud->width);

	PointCloud<PointXYZRGBA>::const_iterator pCloud = cloud.begin();
	PointCloud<PointXYZI>::iterator pLabelCloud = labelCloud->begin();
	Mat_<int>::const_iterator pLabel = label.begin<int>();
	while(pCloud != cloud.end()) {
		pLabelCloud->x = pCloud->x;
		pLabelCloud->y = pCloud->y;
		pLabelCloud->z = pCloud->z;
		pLabelCloud->intensity = *pLabel;
		++pLabel; ++pCloud; ++pLabelCloud;
	}
	labelCloud->sensor_origin_.setZero ();
	labelCloud->sensor_orientation_.w () = 1.0;
	labelCloud->sensor_orientation_.x () = 0.0;
	labelCloud->sensor_orientation_.y () = 0.0;
	labelCloud->sensor_orientation_.z () = 0.0;
}

inline void EstimateNormals(const PointCloud<PointXYZRGBA>::ConstPtr &cloud, PointCloud<PointNormal>::Ptr &normals, bool fill) {
	pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::PointNormal> ne;
	ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
	ne.setMaxDepthChangeFactor(0.02f);
	ne.setNormalSmoothingSize(10.0f);
	ne.setInputCloud(cloud);
	ne.compute(*normals);
	if(fill) {
		PointCloudNormal::iterator p = normals->begin();
		while(p != normals->end()) {
			if(_isnan(p->normal_x))
				p->normal_x = 0;
			if(_isnan(p->normal_y))
				p->normal_y = 0;
			if(_isnan(p->normal_z))
				p->normal_z = 0;
			++p;
		}
	}
}

inline int GetClass(const PointCloudInt &cloud, const Mat &labels, int id) {
	int i, ret = 0;
	int *lookup = new int[NUM_LABELS];
	for(i = 0; i < NUM_LABELS; i++)
		lookup[i] = 0;
	PointCloudInt::const_iterator p = cloud.begin();
	Mat_<int>::const_iterator pL = labels.begin<int>();
	while(p != cloud.end()) {
		if(p->intensity == id)
			lookup[*pL]++;
		++p; ++pL;
	}
	int max = lookup[0], maxLoc = 0;
	for(i = 0; i < NUM_LABELS; i++) {
		if(lookup[i] > max) {
			max = lookup[i];
			maxLoc = i;
		}
	}
	try {
		delete[] lookup;
	} catch(...) {
		cout << "small error here" << endl;
	}
	return maxLoc;
}

inline float HistTotal(LABXYZUVW *hist) {
	float tot = 0.0f;
	for(int k = 0; k < NUM_BINS; k++) {
		tot += hist[k].u;
	}
	return tot;
}

inline void CalcMask(const PointCloudInt &cloud, int id, Mat &mask) {
	PointCloudInt::const_iterator pC = cloud.begin();
	uchar *pM = mask.data;
	while(pC != cloud.end()) {
		if(pC->intensity == id)
			*pM = 255;
		++pM; ++pC;
	}
}

void GetFeatureVectors(Mat &trainData, Classifier &cl, const RegionTree3D &tree, Mat &img, const PointCloudInt &cloud, const Mat &label, const int numImage) {
	//for each top level region, I need to give it a class name.
	int k;
	const int size1 = 14 + 6*NUM_BINS + 3*NUM_BINS_XYZ, size2 = (size1 + NUM_CLUSTERS + 3);
	Mat gImg, desc;
	cvtColor(img, gImg, CV_BGR2GRAY);
	vector<KeyPoint> kp;
	cl.featureDetector->detect(gImg,kp);
	vector<KeyPoint> regionPoints;
	regionPoints.reserve(kp.size());
	vector<Region3D*>::const_iterator p = tree.top_regions.begin();
	//Mat element = getStructuringElement(MORPH_RECT, Size( 2*2 + 1, 2*2+1 ), Point( 2, 2 ) );
	for(int i = 0; i < tree.top_regions.size(); i++, p++) {
		////Calculate mask
		//Mat desc, mask = Mat::zeros(img.size(),CV_8UC1);
		//CalcMask(cloud,(*p)->m_centroid3D.intensity,mask);
		//dilate(mask,mask,element);
		////get features
		//cl.CalculateBOWFeatures(img,mask,desc);
		Mat desc;
		regionPoints.clear();
		vector<KeyPoint>::iterator pK = kp.begin();
		while(pK != kp.end()) {
			PointXYZI p3D = cloud(pK->pt.x,pK->pt.y);
			if(p3D.x >= (*p)->m_min3D.x && p3D.x <= (*p)->m_max3D.x && p3D.y >= (*p)->m_min3D.y && p3D.y <= (*p)->m_max3D.y && p3D.z >= (*p)->m_min3D.z && p3D.z <= (*p)->m_max3D.z)
				regionPoints.push_back(*pK);
			++pK;
		}
		cl.bowDescriptorExtractor->compute(gImg,regionPoints,desc);
		if(desc.empty())
			desc = Mat::zeros(1,NUM_CLUSTERS,CV_32F);
		int id = GetClass(cloud,label,(*p)->m_centroid3D.intensity);
		if(id != 0) {
			Mat vec = Mat(1,size2,CV_32F);
			float *pV = (float*)vec.data;
			*pV++ = float((*p)->m_size);
			*pV++ = (*p)->m_centroid.x;
			*pV++ = (*p)->m_centroid.y;
			*pV++ = (*p)->m_centroid3D.x;
			*pV++ = (*p)->m_centroid3D.y;
			*pV++ = (*p)->m_centroid3D.z;
			float a = ((*p)->m_max3D.x - (*p)->m_min3D.x), b = ((*p)->m_max3D.y - (*p)->m_min3D.y), c = ((*p)->m_max3D.z - (*p)->m_min3D.z);
			*pV++ = (*p)->m_min3D.x;
			*pV++ = (*p)->m_min3D.y;
			*pV++ = (*p)->m_min3D.z;
			*pV++ = (*p)->m_max3D.x;
			*pV++ = (*p)->m_max3D.y;
			*pV++ = (*p)->m_max3D.z;
			*pV++ = sqrt(a*a + c*c);
			*pV++ = b;
			//LABXYZUVW *p1 = (*p)->m_hist;
			//float tot = HistTotal((*p)->m_hist);
			for(k = 0; k < NUM_BINS; k++)
				*pV++ = float((*p)->m_hist[k].a)/(*p)->m_size;
			for(k = 0; k < NUM_BINS; k++)
				*pV++ = float((*p)->m_hist[k].b)/(*p)->m_size;
			for(k = 0; k < NUM_BINS; k++)
				*pV++ = float((*p)->m_hist[k].l)/(*p)->m_size;
			for(k = 0; k < NUM_BINS; k++)
				*pV++ = (*p)->m_hist[k].u/(*p)->m_size;
			for(k = 0; k < NUM_BINS; k++)
				*pV++ = (*p)->m_hist[k].v/(*p)->m_size;
			for(k = 0; k < NUM_BINS; k++)
				*pV++ = (*p)->m_hist[k].w/(*p)->m_size;
			for(k = 0; k < NUM_BINS_XYZ; k++)
				*pV++ = (*p)->m_hist[k].x/(*p)->m_size;
			for(k = 0; k < NUM_BINS_XYZ; k++)
				*pV++ = (*p)->m_hist[k].y/(*p)->m_size;
			for(k = 0; k < NUM_BINS_XYZ; k++)
				*pV++ = (*p)->m_hist[k].z/(*p)->m_size;
			float *pD = (float*)desc.data;
			for(k = 0; k < desc.cols; k++, pD++)
				*pV++ = *pD;
			*pV++ = float((*p)->m_centroid3D.intensity);
			*pV++ = float(numImage);
			*pV++ = float(id);
			trainData.push_back(vec);
		}
	}
}

void GetMatFromRegion(Region3D *reg, Classifier &cl, const PointCloudInt &cloud, vector<KeyPoint> &kp, Mat &img, vector<float> &sample, int sample_size) {
	int k;
	sample.resize(sample_size);
	//Calculate mask
	//Mat desc, mask = Mat::zeros(img.size(),CV_8UC1);
	//CalcMask(cloud,reg->m_centroid3D.intensity,mask);
	////get features
	//cl.CalculateBOWFeatures(img,mask,desc);
	Mat desc;
	vector<KeyPoint> regionPoints;
	regionPoints.reserve(kp.size());
	vector<KeyPoint>::iterator pK = kp.begin();
	while(pK != kp.end()) {
		PointXYZI p3D = cloud(pK->pt.x,pK->pt.y);
		if(p3D.x >= reg->m_min3D.x && p3D.x <= reg->m_max3D.x && p3D.y >= reg->m_min3D.y && p3D.y <= reg->m_max3D.y && p3D.z >= reg->m_min3D.z && p3D.z <= reg->m_max3D.z)
			regionPoints.push_back(*pK);
		++pK;
	}
	cl.bowDescriptorExtractor->compute(img,regionPoints,desc);
	if(desc.empty())
		desc = Mat::zeros(1,NUM_CLUSTERS,CV_32F);
	vector<float>::iterator p = sample.begin();
	*p++ = float(reg->m_size);
	*p++ = reg->m_centroid.x;
	*p++ = reg->m_centroid.y;
	*p++ = reg->m_centroid3D.x;
	*p++ = reg->m_centroid3D.y;
	*p++ = reg->m_centroid3D.z;
	float a = (reg->m_max3D.x - reg->m_min3D.x), b = (reg->m_max3D.y - reg->m_min3D.y), c = (reg->m_max3D.z - reg->m_min3D.z);
	*p++ = reg->m_min3D.x;
	*p++ = reg->m_min3D.y;
	*p++ = reg->m_min3D.z;
	*p++ = reg->m_max3D.x;
	*p++ = reg->m_max3D.y;
	*p++ = reg->m_max3D.z;
	*p++ = sqrt(a*a+c*c);
	*p++ = b;
	for(k = 0; k < NUM_BINS; k++)
		*p++ = reg->m_hist[k].a / reg->m_size;
	for(k = 0; k < NUM_BINS; k++)
		*p++ = reg->m_hist[k].b / reg->m_size;
	for(k = 0; k < NUM_BINS; k++)
		*p++ = reg->m_hist[k].l / reg->m_size;
	for(k = 0; k < NUM_BINS; k++)
		*p++ = reg->m_hist[k].u / reg->m_size;
	for(k = 0; k < NUM_BINS; k++)
		*p++ = reg->m_hist[k].v / reg->m_size;
	for(k = 0; k < NUM_BINS; k++)
		*p++ = reg->m_hist[k].w / reg->m_size;
	for(k = 0; k < NUM_BINS_XYZ; k++)
		*p++ = reg->m_hist[k].x / reg->m_size;
	for(k = 0; k < NUM_BINS_XYZ; k++)
		*p++ = reg->m_hist[k].y / reg->m_size;
	for(k = 0; k < NUM_BINS_XYZ; k++)
		*p++ = reg->m_hist[k].z / reg->m_size;
	float *pD = (float*)desc.data;
	for(k = 0; k < desc.cols; k++, pD++)
		*p++ = *pD;
}

inline void GetMatFromCloud(const PointCloudBgr &cloud, Mat &img) {
	img = Mat(cloud.height,cloud.width,CV_8UC3);
	Mat_<Vec3b>::iterator pI = img.begin<Vec3b>();
	PointCloudBgr::const_iterator pC = cloud.begin();
	while(pC != cloud.end()) {
		(*pI)[0] = pC->b;
		(*pI)[1] = pC->g;
		(*pI)[2] = pC->r;
		++pI; ++pC;
	}
}

inline void GetMatFromCloud(const PointCloudInt &cloud, Mat &img) {
	img = Mat(cloud.height,cloud.width,CV_32S);
	Mat_<int>::iterator pI = img.begin<int>();
	PointCloudInt::const_iterator pC = cloud.begin();
	while(pC != cloud.end()) {
		*pI = pC->intensity;
		++pI; ++pC;
	}
}

inline void GetMatFromCloud(const PointCloudNormal &cloud, Mat &img) {
	img = Mat(cloud.height,cloud.width,CV_8UC3);
	Mat_<Vec3b>::iterator pI = img.begin<Vec3b>();
	PointCloudNormal::const_iterator pC = cloud.begin();
	while(pC != cloud.end()) {
		//scale from -1 to 1
		int red = Round((pC->normal_x + 1) * 127.5f);
		int green = Round((pC->normal_y + 1) * 127.5f);
		int blue = Round((pC->normal_z + 1) * 127.5f);
		*pI = Vec3b(red,green,blue);
		++pI; ++pC;
	}
}

void PseudoColor(const PointCloudInt &in, Mat &out) {
	int min, max;
	MinMax(in, &min, &max);
	int size = max - min;
	Vec3b *colors = (Vec3b *) malloc(size*sizeof(Vec3b));
	Vec3b *pColor = colors;
	for (int i = min; i < max; i++)
	{
		Vec3b color;
		random_rgb(color);
		*pColor++ = color;
	}

	out = Mat::zeros(in.height,in.width,CV_8UC3);
	PointCloudInt::const_iterator pIn = in.begin();
	Mat_<Vec3b>::iterator pOut = out.begin<Vec3b>();
	while(pIn != in.end()) {
		*pOut = colors[int(pIn->intensity) - min];
		pIn++;
		pOut++;
	}
	free(colors);
}

void BuildNYUDataset(string direc, bool matlab) {
	srand(time(NULL));
	PointCloudBgr cloud,segment;
	PointCloudInt labelCloud;
	Mat img, depth, label, trainData;
	boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > normals(new pcl::PointCloud<pcl::PointNormal>);
	Classifier c(direc);
	c.LoadTrainingInd();
	c.load_vocab();
	//open training file
	/*FILE *fp = fopen("features.txt","wb");
	if(fp == NULL)
	throw exception("Couldn't open features file");
	fprintf(fp,"size,cx,cy,c3x,c3y,c3z,minx,miny,minz,maxx,maxy,maxz,xdist,ydist");
	for(int j = 0; j < 9; j++) {
	for(int k = 0; k < (j < 6 ? NUM_BINS : NUM_BINS_XYZ); k++) {
	fprintf(fp,",h%d_%d",j,k);
	}
	}
	fprintf(fp,",frame,class\n");*/
	int count = 0;
	string folder;
	for(int i = 1; i < 1450; i++) {
		if(matlab || i == c.trainingInds.front()) {
			cout << i << endl;
			LoadData(direc,i,img,depth,label);
			CreatePointCloudFromRegisteredNYUData(img,depth,&cloud);
			//CreateLabeledCloudFromNYUPointCloud(cloud,label,&labelCloud);
			int segments = SHGraphSegment(cloud,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],&labelCloud,&segment);
			EstimateNormals(cloud.makeShared(),normals,true);
			RegionTree3D tree;
			tree.Create(cloud,labelCloud,*normals,segments,0);
			tree.PropagateRegionHierarchy(parameters[6]);
			tree.ImplementSegmentation(parameters[7]);

			GetFeatureVectors(trainData,c,tree,img,labelCloud,label,i);
			if(i == c.trainingInds.front()) {
				c.trainingInds.pop_front();
				stringstream num;
				num << "training/" << count << ".flt";
				imwrite_float(num.str().c_str(),trainData);
				count++;
			}
			stringstream num2;
			num2 << "training_all/" << i << ".flt";
			imwrite_float(num2.str().c_str(),trainData);
			stringstream num3;
			num3 << "segments/" << i << ".dep";
			Mat segmentMat, segmentMatColor;
			GetMatFromCloud(labelCloud,segmentMat);
			//PseudoColor(labelCloud,segmentMatColor);
			//imshow("window",segmentMatColor);
			//waitKey();
			imwrite_depth(num3.str().c_str(),segmentMat);

			//release stuff
			segment.clear();
			cloud.clear();
			labelCloud.clear();
			img.release();
			depth.release();
			label.release();
			trainData.release();
			normals->clear();
			tree.top_regions.clear();
			tree.Release();
		}
	}
	FileStorage tot("count.yml", FileStorage::WRITE);
	tot << "count" << count;
	//fclose(fp);
	tot.release();
}

void BuildRFClassifier(string direc) {
	Classifier c(direc);
	c.LoadClassMap(mapFile);
	FileStorage fs("count.yml", FileStorage::READ);
	int i,count;
	fs["count"] >> count;
	fs.release();
	Mat data, train, labels;
	for(i = 0; i < count; i++) {
		Mat tmp;
		stringstream num;
		num << "training/" << i << ".flt";
		tmp = imread_float(num.str().c_str());
		data.push_back(tmp);
	}
	train = data.colRange(0,data.cols-3);
	labels = data.col(data.cols-1);
	labels.convertTo(labels,CV_32S);
	int* pL = (int*)labels.data, *pEnd = pL + labels.rows;
	while(pL != pEnd) {
		*pL = c.classMap[*pL];
		++pL;
	}


	// define all the attributes as numerical
	// alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
	// that can be assigned on a per attribute basis
	Mat var_type = Mat(train.cols + 1, 1, CV_8U );
	var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
	// this is a classification problem (i.e. predict a discrete number of class
	// outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL
	var_type.at<uchar>(train.cols, 0) = CV_VAR_CATEGORICAL;
	//float priors[] = {1,1};
	CvRTParams params = CvRTParams(25, // max depth
		5, // min sample count
		0, // regression accuracy: N/A here
		false, // compute surrogate split, no missing data
		15, // max number of categories (use sub-optimal algorithm for larger numbers)
		nullptr, // the array of priors
		false,  // calculate variable importance
		50,       // number of variables randomly selected at node and used to find the best split(s).
		500,	 // max number of trees in the forest
		0.01f,				// forrest accuracy
		CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
		);

	// train random forest classifier (using training data)
	CvRTrees* rtree = new CvRTrees;

	rtree->train(train, CV_ROW_SAMPLE, labels,
		Mat(), Mat(), var_type, Mat(), params);
	rtree->save("rf.xml");
	delete rtree;
}

void TestRFClassifier(string direc) {
	PointCloudBgr cloud,segment;
	PointCloudInt labelCloud;
	Mat img, depth, label;
	boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > normals(new pcl::PointCloud<pcl::PointNormal>);
	//open training file
	Classifier c(direc);
	c.LoadTestingInd();
	c.LoadClassMap(mapFile);
	c.load_vocab();
	CvRTrees* rtree = new CvRTrees;
	rtree->load("rf.xml");
	Mat conf = Mat::zeros(5,5,CV_32S);
	Mat confClass = Mat::zeros(5,5,CV_32S);
	for(int i = 1; i < 1450; i++) {
		if(i == c.testingInds.front()) {
			c.testingInds.pop_front();
			cout << i << endl;
			LoadData(direc,i,img,depth,label);
			CreatePointCloudFromRegisteredNYUData(img,depth,&cloud);
			//CreateLabeledCloudFromNYUPointCloud(cloud,label,&labelCloud);
			int segments = SHGraphSegment(cloud,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],&labelCloud,&segment);
			EstimateNormals(cloud.makeShared(),normals,false);
			RegionTree3D tree;
			tree.Create(cloud,labelCloud,*normals,segments,0);
			tree.PropagateRegionHierarchy(parameters[6]);
			tree.ImplementSegmentation(parameters[7]);
			/*viewer.removePointCloud("cloud");
			viewer.removePointCloud("original");
			viewer.addPointCloud(segment.makeShared(),"original");
			viewer.addPointCloudNormals<pcl::PointXYZRGBA,pcl::PointNormal>(segment.makeShared(), normals);
			while(1)
			viewer.spinOnce();*/
			int result, feature_len = 14 + 6*NUM_BINS + 3*NUM_BINS_XYZ + NUM_CLUSTERS;
			Mat gImg;
			cvtColor(img, gImg, CV_BGR2GRAY);
			vector<KeyPoint> kp;
			c.featureDetector->detect(gImg,kp);
			//Mat element = getStructuringElement(MORPH_RECT, Size( 2*2 + 1, 2*2+1 ), Point( 2, 2 ) );
			vector<Region3D*>::const_iterator p = tree.top_regions.begin();
			for(int i = 0; i < tree.top_regions.size(); i++, p++) {
				vector<float> sample;
				GetMatFromRegion(*p,c,labelCloud,kp,gImg,sample,feature_len);
				Mat sampleMat = Mat(sample);
				result = Round(rtree->predict(sampleMat));
				int id = GetClass(labelCloud,label,(*p)->m_centroid3D.intensity);
				if(id > 0 && result > 0)
					confClass.at<int>(c.classMap[id],result)++;
				tree.SetBranch(*p,0,result);
			}

			Mat myResult, groundTruth, myResultColor, groundTruthColor, labelColor, segmentMat;
			myResult = Mat(label.rows,label.cols,label.type());
			groundTruth = Mat(label.rows,label.cols,label.type());
			PointCloudInt::iterator pC = labelCloud.begin();
			int *pNewL = (int*)groundTruth.data;
			int *pNewC = (int*)myResult.data;
			int *pL = (int *)label.data;
			while(pC != labelCloud.end()) {
				int newLabel = c.classMap[*pL];
				*pNewL = newLabel;
				*pNewC = pC->intensity;
				/*if(newLabel < 0 || newLabel > 4)
				cout << "label is: " << newLabel << endl;
				if(pC->intensity < 0 || pC->intensity > 4)
				cout << "result is: " << pC->intensity << endl;
				else*/
				if(pC->intensity > 0 && newLabel > 0)
					conf.at<int>(newLabel,pC->intensity)++;
				++pL; ++pC; ++pNewL; ++pNewC;
			}
			/*GetMatFromCloud(segment,segmentMat);
			groundTruth.convertTo(groundTruth,CV_8UC1,63,0);
			myResult.convertTo(myResult,CV_8UC1,63,0);
			label.convertTo(labelColor,CV_8UC1,894,0);
			applyColorMap(groundTruth,groundTruthColor,COLORMAP_JET);
			applyColorMap(myResult,myResultColor,COLORMAP_JET);
			imshow("color",img);
			imshow("original label",labelColor);
			imshow("label",groundTruthColor);
			imshow("result",myResultColor);
			imshow("segment",segmentMat);
			waitKey();*/

			//release stuff
			segmentMat.release();
			myResult.release();
			groundTruth.release();
			myResultColor.release();
			groundTruthColor.release();
			segment.clear();
			cloud.clear();
			labelCloud.clear();
			img.release();
			depth.release();
			label.release();
			normals->clear();
			tree.top_regions.clear();
			tree.Release();
		}
	}

	float tot = 0, result = 0;
	int x,y;
	for(x=0; x<5; x++) {
		for(y=0; y<5; y++) {
			cout << conf.at<int>(x,y) << ", ";
			tot += conf.at<int>(x,y);
			if(x == y)
				result += conf.at<int>(x,y);
		}
		cout << endl;
	}
	cout << "Accuracy: " << (result / tot) << endl;
	cout << endl;
	tot = 0; result = 0;
	for(x=0; x<5; x++) {
		for(y=0; y<5; y++) {
			cout << confClass.at<int>(x,y) << ", ";
			tot += confClass.at<int>(x,y);
			if(x == y)
				result += confClass.at<int>(x,y);
		}
		cout << endl;
	}
	cout << "Class Accuracy: " << (result / tot) << endl;

	delete rtree;
}

void GenerateSegmentMask(const PointCloudInt &labelCloud, int id, Mat &mask) {
	//Here we are going to generate the mask of the segment
	mask = Mat::zeros(Size(labelCloud.width,labelCloud.height), CV_8UC1);
	PointCloudInt::const_iterator p = labelCloud.begin();
	Mat_<uchar>::iterator pO = mask.begin<uchar>();
	while(p != labelCloud.end()) {
		if(p->intensity == id)
			*pO = 255;
		++p; ++pO;
	}
	//Here we are going to grow the mask
	int dilation_size = 5;
	Mat element = getStructuringElement( MORPH_RECT,
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );
	dilate(mask,mask,element);
}

//Let's template this
template<typename T>
void GenerateImageFromMask(const Mat &in, const Mat &mask, Mat &segment) {
	segment = Mat::zeros(in.size(), in.type());
	Mat_<uchar>::const_iterator p = mask.begin<uchar>();
	Mat_<T>::const_iterator pC = in.begin<T>();
	Mat_<T>::iterator pO = segment.begin<T>();
	while(p != mask.end<uchar>()) {
		if(*p == 255)
			*pO = *pC;
		++p; ++pC; ++pO;
	}
}

void GenerateLabelMask(const PointCloudInt &labelCloud, map<int,int> &idLookup, Mat &out) {
	//Here we are going to generate the mask of the segment
	out = Mat::zeros(Size(labelCloud.width,labelCloud.height), CV_32S);
	PointCloudInt::const_iterator p = labelCloud.begin();
	Mat_<int>::iterator pO = out.begin<int>();
	while(p != labelCloud.end()) {
		*pO = idLookup[p->intensity];
		++p; ++pO;
	}
}

void BuildNYUDatasetForCaffe(string direc) {
	srand(time(NULL));
	PointCloudBgr cloud,segment;
	PointCloudInt labelCloud;
	Mat img, depth, label, trainData;
	boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > normals(new pcl::PointCloud<pcl::PointNormal>);
	Classifier c(direc);
	c.LoadTrainingInd();
	c.LoadClassMap(mapFile);

	//Open the training and testing files
	FILE *fpTraining = fopen("training.txt","w");
	if(fpTraining == NULL)
		throw exception("Couldn't open training file");
	FILE *fpTesting = fopen("testing.txt","w");
	if(fpTesting == NULL)
		throw exception("Couldn't open Testing file");

	int count = 0;
	string folder;
	for(int i = 1; i < 1450; i++) {
		cout << i << endl;
		LoadData(direc,i,img,depth,label);
		CreatePointCloudFromRegisteredNYUData(img,depth,&cloud);
		//CreateLabeledCloudFromNYUPointCloud(cloud,label,&labelCloud);
		int segments = SHGraphSegment(cloud,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],&labelCloud,&segment);
		EstimateNormals(cloud.makeShared(),normals,true);
		RegionTree3D tree;
		tree.Create(cloud,labelCloud,*normals,segments,0);
		tree.PropagateRegionHierarchy(parameters[6]);
		tree.ImplementSegmentation(parameters[7]);

		Mat segmentMat, segmentMatColor;
		GetMatFromCloud(labelCloud,segmentMat);
		PseudoColor(labelCloud,segmentMatColor);

		Mat normalsMat;
		GetMatFromCloud(*normals,normalsMat);

		//let's figure out if we are testing or training
		bool training = false;
		if(i == c.trainingInds.front()) {
			c.trainingInds.pop_front();
			training = true;
		}

		//Make a lookup for the old segment ids to new segment names
		std::map<int,int> idLookup;

		vector<Region3D*>::const_iterator p = tree.top_regions.begin();
		for(int i = 0; i < tree.top_regions.size(); i++, p++) {
			//Set up map
			idLookup[(*p)->m_centroid3D.intensity] = count;

			//Get id for label
			int id = GetClass(labelCloud,label,(*p)->m_centroid3D.intensity);
			int mappedId = c.classMap[id];

			//Create segment image for training and save
			Mat mask, segment;
			GenerateSegmentMask(labelCloud, (*p)->m_centroid3D.intensity, mask);

			//Get and save segmented image
			GenerateImageFromMask<Vec3b>(img,mask,segment);
			stringstream imgFileName;
			imgFileName << "segments/" << count << ".png";
			imwrite(imgFileName.str(),segment);

			//Get and save segmented depth
			GenerateImageFromMask<int>(depth,mask,segment);
			stringstream depthFileName;
			depthFileName << "segments_depth/" << count << ".png";
			Mat segmentOut;
			segment.convertTo(segmentOut,CV_16UC1);
			imwrite(depthFileName.str(),segmentOut);

			//Get and save segmented normals
			GenerateImageFromMask<Vec3b>(normalsMat,mask,segment);
			stringstream normalsFileName;
			normalsFileName << "segments_normals/" << count << ".png";
			imwrite(normalsFileName.str(),segment);

			//Write filename and class to training file
			if(training) {
				//I'm training
				fprintf(fpTraining,"%s, %d\n",imgFileName.str().c_str(), mappedId);
			} else {
				//I'm testing
				fprintf(fpTesting,"%s, %d\n",imgFileName.str().c_str(), mappedId);
			}

			++count;
		}

		//Now let's use the map to generate a superpixel label image
		Mat spLabels;
		GenerateLabelMask(labelCloud,idLookup,spLabels);
		stringstream labelsFileName;
		labelsFileName << "labels/" << i << ".dep";
		imwrite_depth(labelsFileName.str().c_str(),spLabels);

		//release stuff
		spLabels.release();
		segmentMat.release();
		segment.clear();
		cloud.clear();
		labelCloud.clear();
		img.release();
		depth.release();
		label.release();
		normals->clear();
		tree.top_regions.clear();
		tree.Release();
	}

	fclose(fpTraining);
	fclose(fpTesting);
}