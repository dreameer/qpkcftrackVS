/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "colorname.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <complex>
#include <cmath>
#include <iostream>


using namespace std;
using namespace cv;


enum MODE 
{
	GRAY = (1<<0),
	CN   = (1<<1),
	CUSTOM = (1<<2)
};

typedef struct Params_{
	double detect_thresh;
	double sigma;
	double lambda;
	double interp_factor;
	double output_sigma_factor;
	bool resize;
	int max_patch_size;
	bool split_coeff;
	bool wrap_kernel;
	int desc_npca;
	int desc_pca;

	//feature compression
	bool compress_feature;
	int compressed_size;
	double pca_learning_rate;
};


class qptracker{

public:
	Params_ params;
	float output_sigma;
	Rect2d roi;
	Mat hann; 	//hann window filter
	Mat hann_cn; //10 dimensional hann-window filter for CN features,

	Mat y,yf; 	// training response and its FFT
	Mat x; 	// observation and its FFT
	Mat k,kf;	// dense gaussian kernel and its FFT
	Mat kf_lambda; // kf+lambda
	Mat new_alphaf, alphaf;	// training coefficients
	Mat new_alphaf_den, alphaf_den; // for splitted training coefficients
	Mat z; // model
	Mat response; // detection result
	Mat old_cov_mtx, proj_mtx; // for feature compression

	// pre-defined Mat variables for optimization of private functions
	Mat spec, spec2;
	std::vector<Mat> layers;
	std::vector<Mat> vxf,vyf,vxyf;
	Mat xy_data,xyf_data;
	Mat data_temp, compress_data;
	std::vector<Mat> layers_pca_data;
	std::vector<Scalar> average_data;
	Mat img_Patch;

	// storage for the extracted features, KRLS model, KRLS compressed model
	Mat X[2],Z[2],Zc[2];

	// storage of the extracted features
	std::vector<Mat> features_pca;
	std::vector<Mat> features_npca;
	std::vector<MODE> descriptors_pca;
	std::vector<MODE> descriptors_npca;

	// optimization variables for updateProjectionMatrix
	Mat data_pca, new_covar,w_data,u_data,vt_data;

	// custom feature extractor
	bool use_custom_extractor_pca;
	bool use_custom_extractor_npca;
	std::vector<void(*)(const Mat img, const Rect roi, Mat& output)> extractor_pca;
	std::vector<void(*)(const Mat img, const Rect roi, Mat& output)> extractor_npca;

	bool resizeImage; // resize the image whenever needed and the patch size is large
	int frame;

	void initParams();
	bool initImpl( const Mat& /*image*/, const Rect2d& boundingBox );
	bool updateImpl( const Mat& image, Rect2d& boundingBox );


	/*
	* KCF functions and vars
	*/
	void qpcreateHanningWindow(OutputArray dest, const cv::Size winSize, const int type) ;
	void inline fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data) ;
	void inline fft2(const Mat src, Mat & dest) ;
	void inline ifft2(const Mat src, Mat & dest) ;
	void inline pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, std::vector<Mat>  & dest, const int flags, const bool conjB=false);
	void inline sumChannels(std::vector<Mat> src, Mat & dest);
	void inline updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  proj_matrix,float pca_rate, int compressed_sz,
		std::vector<Mat> & layers_pca,std::vector<Scalar> & average, Mat pca_data, Mat new_cov, Mat w, Mat u, Mat v);
	void inline compress(const Mat proj_matrix, const Mat src, Mat & dest, Mat & data, Mat & compressed);
	bool getSubWindow(const Mat img, const Rect roi, Mat& feat, Mat& patch, MODE desc = GRAY);
	bool getSubWindow(const Mat img, const Rect roi, Mat& feat, void (*f)(const Mat, const Rect, Mat& ));
	void extractCN(Mat patch_data, Mat & cnFeatures) ;
	void denseGaussKernel(const float sigma, const Mat , const Mat y_data, Mat & k_data,
		std::vector<Mat> & layers_data,std::vector<Mat> & xf_data,std::vector<Mat> & yf_data, std::vector<Mat> xyf_v, Mat xy, Mat xyf );
	void calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data);
	void calcResponse(const Mat alphaf_data, const Mat alphaf_den_data, const Mat kf_data, Mat & response_data, Mat & spec_data, Mat & spec2_data);

	void shiftRows(Mat& mat) ;
	void shiftRows(Mat& mat, int n) ;
	void shiftCols(Mat& mat, int n) ;

};
/*
detect_thresh = 0.5f;     //!<  detection confidence threshold
sigma=0.2f;               //!<  gaussian kernel bandwidth
lambda=0.0001f;           //!<  regularization
interp_factor=0.075f;     //!<  linear interpolation factor for adaptation
output_sigma_factor=1.0f / 16.0f;  //!<  spatial bandwidth (proportional to target)
resize=true;              //!<  activate the resize feature to improve the processing speed
max_patch_size=80*80;     //!<  threshold for the ROI size
split_coeff=true;         //!<  split the training coefficients into two matrices
wrap_kernel=false;        //!<  wrap around the kernel values
desc_npca = GRAY;         //!<  non-compressed descriptors of TrackerKCF::MODE
desc_pca = CN;            //!<  compressed descriptors of TrackerKCF::MODE

//feature compression
compress_feature=true;    //!<  activate the pca method to compress the features
compressed_size=2;        //!<  feature size after compression
pca_learning_rate=0.15f;  //!<  compression learning rate
*
*/
void qptracker::initParams()
{
	params.detect_thresh = 0.5f;
	params.sigma = 0.2f;
	params.lambda = 0.0001f;
	params.interp_factor = 0.075f;
	params.output_sigma_factor = 1.0f / 16.0f;
	params.resize = false;
	params.max_patch_size = 80*80;
	params.split_coeff = true;
	params.wrap_kernel = false;
	params.desc_npca = GRAY;
	params.desc_pca = GRAY;

	params.compress_feature = false; //not used
	params.compressed_size = 1;
	params.pca_learning_rate = 0.15f;
}



/*-------------------------------------
|  implementation of the KCF functions
|-------------------------------------*/

/*
* hann window filter
*/
void qptracker::qpcreateHanningWindow(OutputArray dest, const cv::Size winSize, const int type) {
	CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

	dest.create(winSize, type);
	Mat dst = dest.getMat();

	int rows = dst.rows, cols = dst.cols;

	AutoBuffer<float> _wc(cols);
	float * const wc = (float *)_wc;

	const float coeff0 = 2.0f * (float)CV_PI / (cols - 1);
	const float coeff1 = 2.0f * (float)CV_PI / (rows - 1);
	for(int j = 0; j < cols; j++)
		wc[j] = 0.5f * (1.0f - cos(coeff0 * j));

	if(dst.depth() == CV_32F){
		for(int i = 0; i < rows; i++){
			float* dstData = dst.ptr<float>(i);
			float wr = 0.5f * (1.0f - cos(coeff1 * i));
			for(int j = 0; j < cols; j++)
				dstData[j] = (float)(wr * wc[j]);
		}
	}else{
		for(int i = 0; i < rows; i++){
			double* dstData = dst.ptr<double>(i);
			double wr = 0.5f * (1.0f - cos(coeff1 * i));
			for(int j = 0; j < cols; j++)
				dstData[j] = wr * wc[j];
		}
	}

	// perform batch sqrt for SSE performance gains
	//cv::sqrt(dst, dst); //matlab do not use the square rooted version
}

/*
* simplification of fourier transform function in opencv
*/
void inline qptracker::fft2(const Mat src, Mat & dest)  {
	dft(src,dest,DFT_COMPLEX_OUTPUT);
}

void inline qptracker::fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data) {
	split(src, layers_data);

	for(int i=0;i<src.channels();i++){
		dft(layers_data[i],dest[i],DFT_COMPLEX_OUTPUT);
	}
}

/*
* simplification of inverse fourier transform function in opencv
*/
void inline qptracker::ifft2(const Mat src, Mat & dest) {
	idft(src,dest,DFT_SCALE+DFT_REAL_OUTPUT);
}

/*
* Point-wise multiplication of two Multichannel Mat data
*/
void inline qptracker::pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, std::vector<Mat>  & dest, const int flags, const bool conjB) {
	for(unsigned i=0;i<src1.size();i++){
		mulSpectrums(src1[i], src2[i], dest[i],flags,conjB);
	}
}

/*
* Combines all channels in a multi-channels Mat data into a single channel
*/
void inline qptracker::sumChannels(std::vector<Mat> src, Mat & dest)  {
	dest=src[0].clone();
	for(unsigned i=1;i<src.size();i++){
		dest+=src[i];
	}
}

/*
* obtains the projection matrix using PCA
*/
void inline qptracker::updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  proj_matrix, float pca_rate, int compressed_sz,
											  std::vector<Mat> & layers_pca,std::vector<Scalar> & average, Mat pca_data, Mat new_cov, Mat w, Mat u, Mat vt) {
												  CV_Assert(compressed_sz<=src.channels());

												  split(src,layers_pca);
												  printf("log8.2.1\n");
												  for (int i=0;i<src.channels();i++){
													  average[i]=mean(layers_pca[i]);
													  layers_pca[i]-=average[i];
												  }
												  printf("log8.2.2\n");
												  // calc covariance matrix
												  merge(layers_pca,pca_data);
												  pca_data=pca_data.reshape(1,src.rows*src.cols);
												  printf("log8.2.3\n");
												  bool oclSucceed =false;

												  if (oclSucceed == false) {
													  new_cov=1.0f/(float)(src.rows*src.cols-1)*(pca_data.t()*pca_data);
													  printf("log8.2.3.1\n");
													  if(old_cov.rows==0)old_cov=new_cov.clone();
													  printf("log8.2.3.1.1\n");
													  SVD::compute((1.0f - pca_rate) * old_cov + pca_rate * new_cov, w, u, vt);
													  printf("log8.2.3.2\n");
												  }
												  new_cov=1.0/(float)(src.rows*src.cols-1)*(pca_data.t()*pca_data);
												  printf("log8.2.3.3\n");
												  if(old_cov.rows==0)old_cov=new_cov.clone();
												  printf("log8.2.4\n");
												  // calc PCA
												  SVD::compute((1.0-pca_rate)*old_cov+pca_rate*new_cov, w, u, vt);
												  printf("log8.2.5\n");
												  // extract the projection matrix
												  proj_matrix=u(Rect(0,0,compressed_sz,src.channels())).clone();
												  Mat proj_vars=Mat::eye(compressed_sz,compressed_sz,proj_matrix.type());
												  for(int i=0;i<compressed_sz;i++){
													  proj_vars.at<float>(i,i)=w.at<float>(i);
												  }
												  printf("log8.2.6\n");
												  // update the covariance matrix
												  old_cov=(1.0-pca_rate)*old_cov+pca_rate*proj_matrix*proj_vars*proj_matrix.t();
}

/*
* compress the features
*/
void inline qptracker::compress(const Mat proj_matrix, const Mat src, Mat & dest, Mat & data, Mat & compressed){
	data=src.reshape(1,src.rows*src.cols);
	compressed=data*proj_matrix;
	dest=compressed.reshape(proj_matrix.cols,src.rows).clone();
}

/*
* obtain the patch and apply hann window filter to it
*/
bool qptracker::getSubWindow(const Mat img, const Rect _roi, Mat& feat, Mat& patch, MODE desc)  {

	Rect region=_roi;

	// return false if roi is outside the image
	if((roi & Rect2d(0,0, img.cols, img.rows)) == Rect2d() )
		return false;

	// extract patch inside the image
	if(_roi.x<0){region.x=0;region.width+=_roi.x;}
	if(_roi.y<0){region.y=0;region.height+=_roi.y;}
	if(_roi.x+_roi.width>img.cols)region.width=img.cols-_roi.x;
	if(_roi.y+_roi.height>img.rows)region.height=img.rows-_roi.y;
	if(region.width>img.cols)region.width=img.cols;
	if(region.height>img.rows)region.height=img.rows;

	patch=img(region).clone();

	// add some padding to compensate when the patch is outside image border
	int addTop,addBottom, addLeft, addRight;
	addTop=region.y-_roi.y;
	addBottom=(_roi.height+_roi.y>img.rows?_roi.height+_roi.y-img.rows:0);
	addLeft=region.x-_roi.x;
	addRight=(_roi.width+_roi.x>img.cols?_roi.width+_roi.x-img.cols:0);

	copyMakeBorder(patch,patch,addTop,addBottom,addLeft,addRight,BORDER_REPLICATE);
	if(patch.rows==0 || patch.cols==0)return false;

	// extract the desired descriptors
	switch(desc){
	case CN:
		CV_Assert(img.channels() == 3);
		extractCN(patch,feat);
		feat=feat.mul(hann_cn); // hann window filter
		break;
	default: // GRAY
		if(img.channels()>1)
			cvtColor(patch,feat, CV_BGR2GRAY);
		else
			feat=patch;
		//feat.convertTo(feat,CV_32F);
		feat.convertTo(feat,CV_32F, 1.0/255.0, -0.5);
		//feat=feat/255.0-0.5; // normalize to range -0.5 .. 0.5
		feat=feat.mul(hann); // hann window filter
		break;
	}

	return true;

}

/*
* get feature using external function
*/
bool qptracker::getSubWindow(const Mat img, const Rect _roi, Mat& feat, void (*f)(const Mat, const Rect, Mat& )){

	// return false if roi is outside the image
	if((_roi.x+_roi.width<0)
		||(_roi.y+_roi.height<0)
		||(_roi.x>=img.cols)
		||(_roi.y>=img.rows)
		)return false;

	f(img, _roi, feat);

	if(_roi.width != feat.cols || _roi.height != feat.rows){
		printf("error in customized function of features extractor!\n");
		printf("Rules: roi.width==feat.cols && roi.height = feat.rows \n");
	}

	Mat hann_win;
	std::vector<Mat> _layers;

	for(int i=0;i<feat.channels();i++)
		_layers.push_back(hann);

	merge(_layers, hann_win);

	feat=feat.mul(hann_win); // hann window filter

	return true;
}

/* Convert BGR to ColorNames
*/
void qptracker::extractCN(Mat patch_data, Mat & cnFeatures) {
	Vec3b & pixel = patch_data.at<Vec3b>(0,0);
	unsigned index;

	if(cnFeatures.type() != CV_32FC(10))
		cnFeatures = Mat::zeros(patch_data.rows,patch_data.cols,CV_32FC(10));

	for(int i=0;i<patch_data.rows;i++){
		for(int j=0;j<patch_data.cols;j++){
			pixel=patch_data.at<Vec3b>(i,j);
			index=(unsigned)(floor((float)pixel[2]/8)+32*floor((float)pixel[1]/8)+32*32*floor((float)pixel[0]/8));

			//copy the values
			for(int _k=0;_k<10;_k++){
				cnFeatures.at<Vec<float,10> >(i,j)[_k]=ColorNames[index][_k];
			}
		}
	}

}

/*
*  dense gauss kernel function
*/
void qptracker::denseGaussKernel(const float sigma, const Mat x_data, const Mat y_data, Mat & k_data,
								 std::vector<Mat> & layers_data,std::vector<Mat> & xf_data,std::vector<Mat> & yf_data, std::vector<Mat> xyf_v, Mat xy, Mat xyf )  {
									 double normX, normY;

									 fft2(x_data,xf_data,layers_data);
									 fft2(y_data,yf_data,layers_data);

									 normX=norm(x_data);
									 normX*=normX;
									 normY=norm(y_data);
									 normY*=normY;

									 pixelWiseMult(xf_data,yf_data,xyf_v,0,true);
									 sumChannels(xyf_v,xyf);
									 ifft2(xyf,xyf);

									 if(params.wrap_kernel){
										 shiftRows(xyf, x_data.rows/2);
										 shiftCols(xyf, x_data.cols/2);
									 }

									 //(xx + yy - 2 * xy) / numel(x)
									 xy=(normX+normY-2*xyf)/(x_data.rows*x_data.cols*x_data.channels());

									 // TODO: check wether we really need thresholding or not
									 //threshold(xy,xy,0.0,0.0,THRESH_TOZERO);//max(0, (xx + yy - 2 * xy) / numel(x))
									 for(int i=0;i<xy.rows;i++){
										 for(int j=0;j<xy.cols;j++){
											 if(xy.at<float>(i,j)<0.0)xy.at<float>(i,j)=0.0;
										 }
									 }

									 float sig=-1.0f/(sigma*sigma);
									 xy=sig*xy;
									 exp(xy,k_data);

}

/* CIRCULAR SHIFT Function
* http://stackoverflow.com/questions/10420454/shift-like-matlab-function-rows-or-columns-of-a-matrix-in-opencv
*/
// circular shift one row from up to down
void qptracker::shiftRows(Mat& mat) {

	Mat temp;
	Mat m;
	int _k = (mat.rows-1);
	mat.row(_k).copyTo(temp);
	for(; _k > 0 ; _k-- ) {
		m = mat.row(_k);
		mat.row(_k-1).copyTo(m);
	}
	m = mat.row(0);
	temp.copyTo(m);

}

// circular shift n rows from up to down if n > 0, -n rows from down to up if n < 0
void qptracker::shiftRows(Mat& mat, int n) {
	if( n < 0 ) {
		n = -n;
		flip(mat,mat,0);
		for(int _k=0; _k < n;_k++) {
			shiftRows(mat);
		}
		flip(mat,mat,0);
	}else{
		for(int _k=0; _k < n;_k++) {
			shiftRows(mat);
		}
	}
}

//circular shift n columns from left to right if n > 0, -n columns from right to left if n < 0
void qptracker::shiftCols(Mat& mat, int n) {
	if(n < 0){
		n = -n;
		flip(mat,mat,1);
		transpose(mat,mat);
		shiftRows(mat,n);
		transpose(mat,mat);
		flip(mat,mat,1);
	}else{
		transpose(mat,mat);
		shiftRows(mat,n);
		transpose(mat,mat);
	}
}

/*
* calculate the detection response
*/
void qptracker::calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data) {
	//alpha f--> 2channels ; k --> 1 channel;
	mulSpectrums(alphaf_data,kf_data,spec_data,0,false);
	ifft2(spec_data,response_data);
}

/*
* calculate the detection response for splitted form
*/
void qptracker::calcResponse(const Mat alphaf_data, const Mat _alphaf_den, const Mat kf_data, Mat & response_data, Mat & spec_data, Mat & spec2_data) {

	mulSpectrums(alphaf_data,kf_data,spec_data,0,false);

	//z=(a+bi)/(c+di)=[(ac+bd)+i(bc-ad)]/(c^2+d^2)
	float den;
	for(int i=0;i<kf_data.rows;i++){
		for(int j=0;j<kf_data.cols;j++){
			den=1.0f/(_alphaf_den.at<Vec2f>(i,j)[0]*_alphaf_den.at<Vec2f>(i,j)[0]+_alphaf_den.at<Vec2f>(i,j)[1]*_alphaf_den.at<Vec2f>(i,j)[1]);
			spec2_data.at<Vec2f>(i,j)[0]=
				(spec_data.at<Vec2f>(i,j)[0]*_alphaf_den.at<Vec2f>(i,j)[0]+spec_data.at<Vec2f>(i,j)[1]*_alphaf_den.at<Vec2f>(i,j)[1])*den;
			spec2_data.at<Vec2f>(i,j)[1]=
				(spec_data.at<Vec2f>(i,j)[1]*_alphaf_den.at<Vec2f>(i,j)[0]-spec_data.at<Vec2f>(i,j)[0]*_alphaf_den.at<Vec2f>(i,j)[1])*den;
		}
	}

	ifft2(spec2_data,response_data);
}

/*
* Initialization:
* - creating hann window filter
* - ROI padding
* - creating a gaussian response for the training ground-truth
* - perform FFT to the gaussian response
*/
bool qptracker::initImpl( const Mat& image, const Rect2d& boundingBox )
{
	frame=0;
	roi = boundingBox;

	//calclulate output sigma
	output_sigma=std::sqrt(static_cast<float>(roi.width*roi.height))*params.output_sigma_factor;
	output_sigma=-0.5f/(output_sigma*output_sigma);

	//resize the ROI whenever needed
	if(params.resize && roi.width*roi.height>params.max_patch_size){
		resizeImage=true;
		roi.x/=2.0;
		roi.y/=2.0;
		roi.width/=2.0;
		roi.height/=2.0;
	}

	// add padding to the roi
	roi.x-=roi.width/2;
	roi.y-=roi.height/2;
	roi.width*=2;
	roi.height*=2;

	// initialize the hann window filter
	qpcreateHanningWindow(hann, roi.size(), CV_32F);


	// hann window filter for CN feature
	Mat _layer[] = {hann, hann, hann, hann, hann, hann, hann, hann, hann, hann};
	merge(_layer, 10, hann_cn);

	// create gaussian response
	y=Mat::zeros((int)roi.height,(int)roi.width,CV_32F);
	for(int i=0;i<roi.height;i++){
		for(int j=0;j<roi.width;j++){
			y.at<float>(i,j) =
				static_cast<float>((i-roi.height/2+1)*(i-roi.height/2+1)+(j-roi.width/2+1)*(j-roi.width/2+1));
		}
	}


	y*=(float)output_sigma;
	cv::exp(y,y);

	// perform fourier transfor to the gaussian response
	fft2(y,yf);



	resizeImage = false;
	use_custom_extractor_pca = false;
	use_custom_extractor_npca = false;


	// record the non-compressed descriptors
	if((params.desc_npca & GRAY) == GRAY)descriptors_npca.push_back(GRAY);
	if((params.desc_npca & CN) == CN)descriptors_npca.push_back(CN);
	if(use_custom_extractor_npca)descriptors_npca.push_back(CUSTOM);
	features_npca.resize(descriptors_npca.size());

	// record the compressed descriptors
	if((params.desc_pca & GRAY) == GRAY)descriptors_pca.push_back(GRAY);
	if((params.desc_pca & CN) == CN)descriptors_pca.push_back(CN);
	if(use_custom_extractor_pca)descriptors_pca.push_back(CUSTOM);
	features_pca.resize(descriptors_pca.size());

	// accept only the available descriptor modes
	CV_Assert(
		(params.desc_pca & GRAY) == GRAY
		|| (params.desc_npca & GRAY) == GRAY
		|| (params.desc_pca & CN) == CN
		|| (params.desc_npca & CN) == CN

		|| use_custom_extractor_pca
		|| use_custom_extractor_npca
		);

	//return true only if roi has intersection with the image
	if((roi & Rect2d(0,0, resizeImage ? image.cols / 2 : image.cols,
		resizeImage ? image.rows / 2 : image.rows)) == Rect2d())
		return false;

	return true;
}

/*
* Main part of the KCF algorithm
*/
bool qptracker::updateImpl( const Mat& image, Rect2d& boundingBox )
{
	double minVal, maxVal;	// min-max response
	Point minLoc,maxLoc;	// min-max location

	Mat img=image.clone();
	// check the channels of the input image, grayscale is preferred
	CV_Assert(img.channels() == 1 || img.channels() == 3);

	// resize the image whenever needed
	if(resizeImage)resize(img,img,Size(img.cols/2,img.rows/2));

	printf("log1\n");
	// detection part
	if(frame>0)
	{
		// extract and pre-process the patch



		// get non compressed descriptors
		for(unsigned i=0;i<descriptors_npca.size()-extractor_npca.size();i++){
			if(!getSubWindow(img,roi, features_npca[i], img_Patch, descriptors_npca[i]))return false;
		}
		//get non-compressed custom descriptors
		for(unsigned i=0,j=(unsigned)(descriptors_npca.size()-extractor_npca.size());i<extractor_npca.size();i++,j++){
			if(!getSubWindow(img,roi, features_npca[j], extractor_npca[i]))return false;
		}
		if(features_npca.size()>0)merge(features_npca,X[1]);



		printf("log2\n");


		// get compressed descriptors
		for(unsigned i=0;i<descriptors_pca.size()-extractor_pca.size();i++){
			if(!getSubWindow(img,roi, features_pca[i], img_Patch, descriptors_pca[i]))return false;
		}
		//get compressed custom descriptors
		for(unsigned i=0,j=(unsigned)(descriptors_pca.size()-extractor_pca.size());i<extractor_pca.size();i++,j++){
			if(!getSubWindow(img,roi, features_pca[j], extractor_pca[i]))return false;
		}
		if(features_pca.size()>0)merge(features_pca,X[0]);


		printf("log3\n");


		//compress the features and the KRSL model
		if(params.desc_pca !=0){
			compress(proj_mtx,X[0],X[0],data_temp,compress_data);
			compress(proj_mtx,Z[0],Zc[0],data_temp,compress_data);
		}
		// copy the compressed KRLS model
		Zc[1] = Z[1];

		// merge all features
		if(features_npca.size()==0){
			x = X[0];
			z = Zc[0];
		}else if(features_pca.size()==0){
			x = X[1];
			z = Z[1];
		}else{
			merge(X,2,x);
			merge(Zc,2,z);
		}


		printf("log4\n");


		//compute the gaussian kernel
		denseGaussKernel(params.sigma,x,z,k,layers,vxf,vyf,vxyf,xy_data,xyf_data);

		// compute the fourier transform of the kernel
		fft2(k,kf);

		if(frame==1)spec2=Mat_<Vec2f >(kf.rows, kf.cols);

		// calculate filter response
		if(params.split_coeff)
			calcResponse(alphaf,alphaf_den,kf,response, spec, spec2);
		else
			calcResponse(alphaf,kf,response, spec);

		// extract the maximum response
		minMaxLoc( response, &minVal, &maxVal, &minLoc, &maxLoc );
		if (maxVal < params.detect_thresh)
		{
			return false;
		}
		roi.x+=(maxLoc.x-roi.width/2+1);
		roi.y+=(maxLoc.y-roi.height/2+1);

	}

	printf("log5\n");

	// update the bounding box
	boundingBox.x=(resizeImage?roi.x*2:roi.x)+(resizeImage?roi.width*2:roi.width)/4;
	boundingBox.y=(resizeImage?roi.y*2:roi.y)+(resizeImage?roi.height*2:roi.height)/4;
	boundingBox.width = (resizeImage?roi.width*2:roi.width)/2;
	boundingBox.height = (resizeImage?roi.height*2:roi.height)/2;


	printf("log6\n");


	// extract the patch for learning purpose
	// get non compressed descriptors
	for(unsigned i=0;i<descriptors_npca.size()-extractor_npca.size();i++){
		if(!getSubWindow(img,roi, features_npca[i], img_Patch, descriptors_npca[i]))return false;
	}
	//get non-compressed custom descriptors
	for(unsigned i=0,j=(unsigned)(descriptors_npca.size()-extractor_npca.size());i<extractor_npca.size();i++,j++){
		if(!getSubWindow(img,roi, features_npca[j], extractor_npca[i]))return false;
	}
	if(features_npca.size()>0)merge(features_npca,X[1]);

	// get compressed descriptors
	for(unsigned i=0;i<descriptors_pca.size()-extractor_pca.size();i++){
		if(!getSubWindow(img,roi, features_pca[i], img_Patch, descriptors_pca[i]))return false;
	}
	//get compressed custom descriptors
	for(unsigned i=0,j=(unsigned)(descriptors_pca.size()-extractor_pca.size());i<extractor_pca.size();i++,j++){
		if(!getSubWindow(img,roi, features_pca[j], extractor_pca[i]))return false;
	}
	if(features_pca.size()>0)merge(features_pca,X[0]);



	printf("log7\n");



	//update the training data
	if(frame==0){
		Z[0] = X[0].clone();
		Z[1] = X[1].clone();
	}else{
		Z[0]=(1.0-params.interp_factor)*Z[0]+params.interp_factor*X[0];
		Z[1]=(1.0-params.interp_factor)*Z[1]+params.interp_factor*X[1];
	}


	printf("log8\n");


	if(params.desc_pca !=0 || use_custom_extractor_pca){
		// initialize the vector of Mat variables
		if(frame==0){
			printf("log8.1\n");
			layers_pca_data.resize(Z[0].channels());
			average_data.resize(Z[0].channels());
		}
		printf("log8.2\n");
		// feature compression
		updateProjectionMatrix(Z[0],old_cov_mtx,proj_mtx,params.pca_learning_rate,params.compressed_size,layers_pca_data,average_data,data_pca, new_covar,w_data,u_data,vt_data);
		printf("log8.3\n");
		compress(proj_mtx,X[0],X[0],data_temp,compress_data);
	}

	printf("log8.5\n");

	// merge all features
	if(features_npca.size()==0)
		x = X[0];
	else if(features_pca.size()==0)
		x = X[1];
	else
		merge(X,2,x);

	// initialize some required Mat variables
	if(frame==0){
		layers.resize(x.channels());
		vxf.resize(x.channels());
		vyf.resize(x.channels());
		vxyf.resize(vyf.size());
		new_alphaf=Mat_<Vec2f >(yf.rows, yf.cols);
	}


	printf("log9\n");


	// Kernel Regularized Least-Squares, calculate alphas
	denseGaussKernel(params.sigma,x,x,k,layers,vxf,vyf,vxyf,xy_data,xyf_data);

	// compute the fourier transform of the kernel and add a small value
	fft2(k,kf);
	kf_lambda=kf+params.lambda;

	float den;
	if(params.split_coeff){
		mulSpectrums(yf,kf,new_alphaf,0);
		mulSpectrums(kf,kf_lambda,new_alphaf_den,0);
	}else{
		for(int i=0;i<yf.rows;i++){
			for(int j=0;j<yf.cols;j++){
				den = 1.0f/(kf_lambda.at<Vec2f>(i,j)[0]*kf_lambda.at<Vec2f>(i,j)[0]+kf_lambda.at<Vec2f>(i,j)[1]*kf_lambda.at<Vec2f>(i,j)[1]);

				new_alphaf.at<Vec2f>(i,j)[0]=
					(yf.at<Vec2f>(i,j)[0]*kf_lambda.at<Vec2f>(i,j)[0]+yf.at<Vec2f>(i,j)[1]*kf_lambda.at<Vec2f>(i,j)[1])*den;
				new_alphaf.at<Vec2f>(i,j)[1]=
					(yf.at<Vec2f>(i,j)[1]*kf_lambda.at<Vec2f>(i,j)[0]-yf.at<Vec2f>(i,j)[0]*kf_lambda.at<Vec2f>(i,j)[1])*den;
			}
		}
	}

	printf("log10\n");


	// update the RLS model
	if(frame==0){
		alphaf=new_alphaf.clone();
		if(params.split_coeff)alphaf_den=new_alphaf_den.clone();
	}else{
		alphaf=(1.0-params.interp_factor)*alphaf+params.interp_factor*new_alphaf;
		if(params.split_coeff)alphaf_den=(1.0-params.interp_factor)*alphaf_den+params.interp_factor*new_alphaf_den;
	}


	frame++;
	return true;
}



int main(){
	cv::VideoCapture inputcamera(0);
	//inputcamera.set(CAP_PROP_FRAME_WIDTH,1280);
	//inputcamera.set(CAP_PROP_FRAME_HEIGHT,720);
	Mat frame,gray;
	Rect2d object_rect, init_rect, center_rect;
	char cmd;
	bool maincontrol,initstatus,re_init;


	inputcamera >> frame;
	center_rect = Rect(frame.cols * 0.40, frame.rows * 0.45,frame.cols * 0.2, frame.rows * 0.1);
	init_rect = center_rect;
	object_rect = init_rect;

	maincontrol = true;
	initstatus = false;
	re_init = false;
	cmd = 'a';
	Ptr<qptracker> tracker;

	while(maincontrol){
		int64 start = cv::getTickCount();
		inputcamera >> frame;
		cvtColor(frame,gray,CV_BGR2GRAY);


		switch(cmd){
		case 'e':maincontrol = false;break;
		case 'q':re_init = true;break;
		case 'w':initstatus = false;break;
		default:break;
		}


		if(re_init){
			cout<<"re init"<<endl;
			object_rect = init_rect;
			tracker = new qptracker;
			tracker->initParams();
			initstatus = tracker->initImpl(gray,object_rect);
			re_init = false;
		}

		if(initstatus)
		{
			cout<<"initstatus true"<<endl;
			if(tracker->updateImpl(gray,object_rect)){
				cout<<"update impl true"<<endl;
				init_rect = object_rect;
				rectangle(frame,object_rect,Scalar(255,0,0),2,1);
			}else{
				cout<<"updateimpl false"<<endl;
				rectangle(frame,object_rect,Scalar(0,255,0),2,1);
			}
		}
		else{
			rectangle(frame,init_rect,Scalar(0,0,255),2,1);
		}

		double fps = cv::getTickFrequency() / (cv::getTickCount()-start);

		std::ostringstream stm;
		stm << fps;
		string frameinfo = " fps:" + stm.str();
		putText(frame, frameinfo, Point(10, 40),FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
		imshow("frame",frame);
		cmd = (char)waitKey(1);
	}
	return 0;
}
