#include "KKA.h"
//#include "MAD.h"
#include <time.h>

using namespace std;
using namespace cv;

int main() {
	cv::VideoCapture cap;
	cap.open("video.mp4");

	if (cap.isOpened() == 0) {
		cout << "The video file cannot be opened." << endl;
		return -1;
	}

	Mat R = imread("r.png", IMREAD_COLOR);
	Mat R_yrb, R_y, C_yrb, C_y, trying;

	cvtColor(R, R_yrb, COLOR_BGR2YCrCb);
	extractChannel(R_yrb, R_y, 0);

	cv::Mat frame;
	cap >> frame;
	cvtColor(frame, C_yrb, COLOR_BGR2YCrCb);
	extractChannel(C_yrb, C_y, 0);

	clock_t tStart = clock();
	clock_t tEnd = clock();
	clock_t tFrame;
	clock_t tBegin = clock();

	vector<double> coord = KKA(C_y, R_y); //KKA
	//vector<double> coord = MAD(C_y, R_y); //MAD
	cout << "\nTime taken:\t" << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;
	Rect rect(coord[1], coord[0], R_y.cols, R_y.rows);
	rectangle(frame, rect, Scalar(0, 255, 0));

	imshow("Tracking", frame);

	while (true) {
		tFrame = clock();
		cap >> frame;
		if (frame.empty()) break; 

		if ((double)(tEnd - tStart) >= 100) {

			cvtColor(frame, C_yrb, COLOR_BGR2YCrCb);
			extractChannel(C_yrb, C_y, 0);

			if (checkCorrelationValue(coord[2])) {
				coord = trackingKKA(C_y, R_y, coord); //KKA
				//coord = trackingMAD(C_y, R_y, coord); //MAD
				rect = Rect(coord[1], coord[0], R_y.cols, R_y.rows);
				tStart = clock();
			}
			else {
				coord = trackingKKA(C_y, R_y, coord); //KKA
				//coord = trackingMAD(C_y, R_y, coord); //MAD
				rect = Rect(coord[1], coord[0], R_y.cols, R_y.rows);
				tStart = clock();

				Mat R_new;
				R_new = frame(rect);
				R = R_new;
				cvtColor(R, R_yrb, COLOR_BGR2YCrCb);
				extractChannel(R_yrb, R_y, 0);
			}
		}

		rectangle(frame, rect, Scalar(0, 255, 0));
		cv::imshow("Tracking", frame);
		tEnd = clock();
		if ((char)cv::waitKey(1) >= 0) break;
		cout << "\nFrame duration:\t" << (double)(clock() - tFrame) / CLOCKS_PER_SEC << endl;
	}

	cout << "\nAlgoritm duration:\t" << (double)(clock() - tBegin) / CLOCKS_PER_SEC << endl;

	waitKey(0);
	return 0;
}
