#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

vector<double> findMin(Mat K, int i0, int j0) {

	double min = K.at<double>(0, 0);
	int i_min = i0, j_min = j0;

	for (int i = 0; i < K.rows; i++)
		for (int j = 0; j < K.cols; j++) {
			if (min > K.at<double>(i, j)) {
				min = K.at<double>(i, j);
				i_min = i + i0;
				j_min = j + j0;
			}
		}
	return { (double)i_min, (double)j_min, min };
}

vector<double> MAD(Mat C_y, Mat R_y) {

	Mat K(C_y.rows - R_y.rows, C_y.cols - R_y.cols, CV_64F);
	double sum = 0;

	for (int di = 0; di < (C_y.rows - R_y.rows); di++)
		for (int dj = 0; dj < (C_y.cols - R_y.cols); dj++) {
			for (int i = 0; i < R_y.rows; i += 2)
				for (int j = 0; j < R_y.cols; j += 2) {
					sum += abs(R_y.at<uchar>(i, j) - C_y.at<uchar>(i + di, j + dj));
				}
			K.at<double>(di, dj) = sum / (C_y.cols * C_y.rows);
			sum = 0;
		}

	return findMin(K, 0, 0);
}

bool checkCorrelationValue(double K) {
	if (K > 0.9) return true;
	else return false;
}

vector<double> trackingMAD(Mat C_y, Mat R_y, vector<double> coord) {

	Mat K(3 * R_y.rows, 3 * R_y.cols, CV_64F);
	double sum = 0;

	int di0 = (int)coord[0] - ceil(1.5 * R_y.rows); di0 < 0 ? 0 : di0;
	int dik = (int)coord[0] + ceil(1.5 * R_y.rows); dik >= C_y.rows ? C_y.rows : dik;
	int dj0 = (int)coord[1] - ceil(1.5 * R_y.cols); dj0 < 0 ? 0 : dj0;
	int djk = (int)coord[1] + ceil(1.5 * R_y.cols); djk >= C_y.cols ? C_y.cols : djk;

	for (int di = di0; di < dik; di++)
		for (int dj = dj0; dj < djk; dj++) {
			for (int i = 0; i < R_y.rows; i += 2)
				for (int j = 0; j < R_y.cols; j += 2) {
					sum += abs(R_y.at<uchar>(i, j) - C_y.at<uchar>(i + di, j + dj));
				}
			K.at<double>(di - di0, dj - dj0) = sum / (C_y.cols * C_y.rows);
			sum = 0;
		}

	return findMin(K, di0, dj0);
}
