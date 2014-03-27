#include "TestVideoSegmentation.h"

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
			out = Mat(width,height,CV_32S);
			int *p = (int*)out.data;
			fread(p,sizeof(int),width*height,fp);
		} else {
			//fscanf(fp,"%i,%i,",&width,&height);
			out = Mat(width,height,CV_32S);
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
	ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
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
	delete[] lookup;
	return maxLoc;
}

void GetFeatureVectors(FILE *fp, const RegionTree3D &tree, PointCloudInt &cloud, const Mat &label) {
	//for each top level region, I need to give it a class name.
	int k;
	vector<Region3D*>::const_iterator p = tree.top_regions.begin();
	for(int i = 0; i < tree.top_regions.size(); i++, p++) {
		int id = GetClass(cloud,label,(*p)->m_centroid3D.intensity);
		fprintf(fp,"%d,%f,%f,%f,%f,%f",(*p)->m_size,(*p)->m_centroid.x,(*p)->m_centroid.y,(*p)->m_centroid3D.x,(*p)->m_centroid3D.y,(*p)->m_centroid3D.z);
		//LABXYZUVW *p1 = (*p)->m_hist;
		for(k = 0; k < NUM_BINS; k++)
			fprintf(fp,",%d",(*p)->m_hist[k].a);
		for(k = 0; k < NUM_BINS; k++)
			fprintf(fp,",%d",(*p)->m_hist[k].b);
		for(k = 0; k < NUM_BINS; k++)
			fprintf(fp,",%d",(*p)->m_hist[k].l);
		for(k = 0; k < NUM_BINS; k++)
			fprintf(fp,",%f",(*p)->m_hist[k].u);
		for(k = 0; k < NUM_BINS; k++)
			fprintf(fp,",%f",(*p)->m_hist[k].v);
		for(k = 0; k < NUM_BINS; k++)
			fprintf(fp,",%f",(*p)->m_hist[k].w);
		for(k = 0; k < NUM_BINS_XYZ; k++)
			fprintf(fp,",%f",(*p)->m_hist[k].x);
		for(k = 0; k < NUM_BINS_XYZ; k++)
			fprintf(fp,",%f",(*p)->m_hist[k].y);
		for(k = 0; k < NUM_BINS_XYZ; k++)
			fprintf(fp,",%f",(*p)->m_hist[k].z);
		fprintf(fp,",%d\n",id);
	}
}

void BuildNYUDataset(string direc) {
	srand(time(NULL));
	PointCloudBgr cloud,segment;
	PointCloudInt labelCloud;
	Mat img, depth, label;
	boost::shared_ptr<pcl::PointCloud<pcl::PointNormal> > normals(new pcl::PointCloud<pcl::PointNormal>);
	FILE *fp = fopen("features.txt","w");
	if(fp == NULL)
		throw exception("Couldn't open features file");
	fprintf(fp,"size,cx,cy,c3x,c3y,c3z");
	for(int j = 0; j < 9; j++) {
		for(int k = 0; k < (j < 6 ? NUM_BINS : NUM_BINS_XYZ); k++) {
			fprintf(fp,",h%d_%d",j,k);
		}
	}
	fprintf(fp,",class\n");
	pcl::visualization::PCLVisualizer viewer("New viewer");
	for(int i = 1; i < 1450; i++) {
		stringstream num;
		num << i;
		img = imread(string(direc + "rgb\\" + num.str() + ".bmp"));
		depth = imread_depth(string(direc + "depth\\" + num.str() + ".dep").c_str(),true);
		label = imread_depth(string(direc + "labels\\" + num.str() + ".dep").c_str(),true);
		CreatePointCloudFromRegisteredNYUData(img,depth,&cloud);
		//CreateLabeledCloudFromNYUPointCloud(cloud,label,&labelCloud);
		int segments = SHGraphSegment(cloud,2.5f,400.0f,100,0.8f,100.0f,75,&labelCloud,&segment);
		EstimateNormals(cloud.makeShared(),normals,true);
		RegionTree3D tree;
		tree.Create(cloud,labelCloud,*normals,segments,0);
		tree.PropagateRegionHierarchy(75);
		tree.ImplementSegmentation(0.4f);
		viewer.removePointCloud("cloud");
		viewer.removePointCloud("original");
		viewer.addPointCloud(segment.makeShared(),"original");
		viewer.addPointCloudNormals<pcl::PointXYZRGBA,pcl::PointNormal>(segment.makeShared(), normals);
		while(1)
			viewer.spinOnce();
		GetFeatureVectors(fp,tree,labelCloud,label);
		//release stuff
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
	fclose(fp);
}