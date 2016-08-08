#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"


#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\cudaobjdetect.hpp"
#include "opencv2\cudaimgproc.hpp"
#include "opencv2\cudawarping.hpp"
//cuda codes
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudabgsegm.hpp"

using namespace cv;
using namespace std;

static void download(const cuda::GpuMat& d_mat, vector< Point2f>& vec);
static void download(const cuda::GpuMat& d_mat, vector< uchar>& vec);
static void drawArrows(Mat& frame, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, Scalar line_color = Scalar(0, 0, 255));

void main()
{
	//variable
	cuda::GpuMat GpuImg, rGpuImg_Bgray;
	cuda::GpuMat oldGpuImg_Agray;

	//video
	Mat img, dImg_rg, dimg;
	printf("Starting Video...\n");
	VideoCapture cap(0);
	if (!cap.isOpened())  // check if succeeded to connect to the camera
		CV_Assert("Cam open failed");

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);


	cap >> img;
	if (img.empty())
		return;
	printf("Got first frame...\n");
	//scale
	double scale = 1.0;

	printf("Got first frame on GPU...\n");
	//first gpumat
	GpuImg.upload(img);
	cuda::resize(GpuImg, oldGpuImg_Agray, Size(GpuImg.cols * scale, GpuImg.rows * scale));
	cuda::cvtColor(oldGpuImg_Agray, oldGpuImg_Agray, CV_BGR2GRAY);

	printf("Pushed on GPU...\n");
	cuda::GpuMat d_prevPts;
	cuda::GpuMat d_nextPts;
	cuda::GpuMat d_status;
	Ptr< cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(oldGpuImg_Agray.type(), 4000, 0.01, 0);
	//opticla flow
	Ptr< cuda::SparsePyrLKOpticalFlow> d_pyrLK = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);

	unsigned long Atime, Btime;
	float TakeTime;
	printf("Ready for first run...\n");

	while (1)
	{
		Atime = getTickCount();

		cap >> img;  
		if (img.empty())
			break;

		//get image
		GpuImg.upload(img);
		cuda::resize(GpuImg, rGpuImg_Bgray, Size(GpuImg.cols * scale, GpuImg.rows * scale));
		rGpuImg_Bgray.download(dimg);
		cuda::cvtColor(rGpuImg_Bgray, rGpuImg_Bgray, CV_BGR2GRAY);
		rGpuImg_Bgray.download(dImg_rg);


		//A,B image  
		//oldGpuImg_Agray;
		//rGpuImg_Bgray;

		//feature
		detector->detect(oldGpuImg_Agray, d_prevPts);
		d_pyrLK->calc(oldGpuImg_Agray, rGpuImg_Bgray, d_prevPts, d_nextPts, d_status);


		//old
		oldGpuImg_Agray = rGpuImg_Bgray;


		// Draw arrows
		vector< Point2f> prevPts(d_prevPts.cols);
		download(d_prevPts, prevPts);

		vector< Point2f> nextPts(d_nextPts.cols);
		download(d_nextPts, nextPts);

		vector< uchar> status(d_status.cols);
		download(d_status, status);

		drawArrows(dimg, prevPts, nextPts, status, Scalar(0, 255, 0));

		//show
		imshow("PyrLK [Sparse]", dimg);
		imshow("origin", dImg_rg);
		if (waitKey(1)>0)
			break;


		Btime = getTickCount();
		TakeTime = (Btime - Atime) / getTickFrequency();
		printf("%lf sec / %lf fps \n", TakeTime, 1 / TakeTime);
	}
	d_pyrLK.release();
	detector.release();
}



static void download(const cuda::GpuMat& d_mat, vector< uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

static void download(const cuda::GpuMat& d_mat, vector< Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

static void drawArrows(Mat& frame, const vector< Point2f>& prevPts, const vector< Point2f>& nextPts, const vector< uchar>& status, Scalar line_color)
{
	for (size_t i = 0; i < prevPts.size(); ++i)
	{
		if (status[i])
		{
			int line_thickness = 1;

			Point p = prevPts[i];
			Point q = nextPts[i];

			double angle = atan2((double)p.y - q.y, (double)p.x - q.x);

			double hypotenuse = sqrt((double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x));

			if (hypotenuse <  1.0)
				continue;

			// Here we lengthen the arrow by a factor of three.
			q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
			q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

			// Now we draw the main line of the arrow.
			line(frame, p, q, line_color, line_thickness);

			// Now draw the tips of the arrow. I do some scaling so that the
			// tips look proportional to the main line of the arrow.

			p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
			p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
			line(frame, p, q, line_color, line_thickness);

			p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
			p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
			line(frame, p, q, line_color, line_thickness);
		}
	}
}

//int main(int argc, char* argv[])
//{
//	VideoCapture cap(0); // open the default camera
//	if (!cap.isOpened())  // check if we succeeded
//		return -1;
//
//	Mat edges;
//	namedWindow("Result", 1);
//
//	for (;;)
//	{
//		Mat frame;
//		cap >> frame; // get a new frame from camera
//		//cvtColor(frame, edges, CV_BGR2GRAY);
//		//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
//		//Canny(edges, edges, 0, 30, 3);
//		//imshow("edges", edges);
//		try
//		{
//			//cv::Mat src_host = cv::imread("E:/temp1.jpg", cv::IMREAD_GRAYSCALE);
//			cv::Mat src_host = edges;
//			cv::cuda::GpuMat dst, src;
//			src.upload(src_host);
//			cv::cuda::HoughLinesDetector
//			cv::cuda::
//			
//			cv::Mat result_host(dst);
//			cv::imshow("Result", result_host);
//
//			//Ptr<ORB> orb = ORB::create();
//			//std::vector<KeyPoint> keypoints;
//			//Mat descriptors;
//			//orb->detectAndCompute(frame, Mat(), keypoints, descriptors);
//			//Ptr<cuda::ORB> d_orb = cuda::ORB::create();
//			//cuda::GpuMat d_src(frame);
//			//cuda::GpuMat d_keypoints;
//			//cuda::GpuMat d_descriptors;
//			//d_orb->detectAndComputeAsync(d_src, cuda::GpuMat(), d_keypoints, d_descriptors);
//		}
//		catch (const cv::Exception& ex)
//		{
//			std::cout << "Error: " << ex.what() << std::endl;
//		}
//
//
//		if (waitKey(1) >= 0) break;
//	}
//
//}