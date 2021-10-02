#include <iostream>
#include <ctime>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef struct NCCinformation {
	int x = 0;			// 定义匹配点X位置
	int y = 0;			// 定义匹配点Y位置
	double value = 0;	// NCC匹配值
	double theta = 0;	// 旋转角度值
} NCCinfo;

void RotateArbitraryAngle(Mat& src, Mat& dst, float angle) {
	double radian = (double)(angle / 180.0 * CV_PI);

	// 填充图像
	int maxBorder = (int)(max(src.cols, src.rows) * 1.414);
	int dx = (maxBorder - src.cols) / 2;
	int dy = (maxBorder - src.rows) / 2;
	copyMakeBorder(src, dst, dy, dy, dx, dx, BORDER_CONSTANT);

	// 旋转
	Point2f center((double)(dst.cols / 2), (double)(dst.rows / 2));
	Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);
	warpAffine(dst, dst, affine_matrix, dst.size());

	// 计算图像旋转之后包含图像的最大的矩形
	double sinVal = abs(sin(radian));
	double cosVal = abs(cos(radian));
	Size targetSize((int)(src.cols * cosVal + src.rows * sinVal), (int)(src.cols * sinVal + src.rows * cosVal));

	// 剪掉多余边框
	int x = (dst.cols - targetSize.width) / 2;
	int y = (dst.rows - targetSize.height) / 2;
	Rect rect(x, y, targetSize.width, targetSize.height);
	dst = Mat(dst, rect);
}


Mat ImagePyramid(Mat sourceimg, int levelnum) {
	Mat temp, temp1, baselevel, finallevel;
	temp = sourceimg.clone();
	baselevel = sourceimg.clone();

	for (int i = 1; i < levelnum; i++) {
		GaussianBlur(temp, temp1, Size(5, 5), 0);
		resize(temp, temp1, Size(temp.cols / 2, temp.rows / 2), 0, 0, INTER_CUBIC);
		temp = temp1.clone();
	}
	finallevel = temp1.clone();
	return finallevel;
}


double ImageAverage(Mat img) {
	Mat temp = img.clone();
	int Square = 0;
	double sum = 0;
	for (int i = 0; i < temp.rows; i++) {
		for (int j = 0; j < temp.cols; j++) {
			if (0 == temp.at<uchar>(i, j))
				continue;
			sum += temp.at<uchar>(i, j);
			Square += 1;
		}
	}
	double average = sum / Square;
	return average;
}


double ImageStandardDeviation(Mat img, double average) {
	Mat temp = img.clone();
	int Square = 0;
	double sum = 0;
	for (int i = 0; i < temp.rows; i++) {
		for (int j = 0; j < temp.cols; j++) {
			if (0 == temp.at<uchar>(i, j))
				continue;
			sum += pow((temp.at<uchar>(i, j) - average), 2);
			Square += 1;
		}
	}
	double sd = sqrt(sum / Square);
	return sd;
}


NCCinfo NormalizedCrossCorrelation(Mat sourceimg, Mat templateimg) {
	NCCinfo info;
	Mat src = sourceimg.clone();
	Mat Tmplate = templateimg.clone();

	// 获取两张图像的长和宽
	int srcrows = src.rows;
	int srccols = src.cols;
	int Tmprows = Tmplate.rows;
	int Tmpcols = Tmplate.cols;
	int TmpSquare = 0;

	// 计算模板图片的灰度平均值和标准差
	double TmplateAverage = ImageAverage(Tmplate);
	double Tmpstandarddeviation = ImageStandardDeviation(Tmplate, TmplateAverage);

	double SrcAverage;
	double Srcstandarddeviation;

	// 计算NCC图像
	long double** NCC;
	NCC = new long double* [srcrows];
	for (int j = 0; j < srcrows; j++) {
		NCC[j] = new long double[srccols];
	}
	for (int i = 0; i < srcrows; i++) {
		for (int j = 0; j < srccols; j++) {
			NCC[i][j] = 0;
		}
	}

	double sum = 0;
	Mat temp;
	for (int i = 0; i < srcrows - Tmprows; i++) {
		for (int j = 0; j < srccols - Tmpcols; j++) {
			TmpSquare = 0;
			temp = src(Rect(j, i, Tmpcols, Tmprows));
			SrcAverage = ImageAverage(temp);
			Srcstandarddeviation = ImageStandardDeviation(temp, SrcAverage);
			for (int k = 0; k < Tmprows; k++) {
				for (int h = 0; h < Tmpcols; h++) {
					if (0 == Tmplate.at<uchar>(k, h))
						continue;
					sum += ((Tmplate.at<uchar>(k, h) - TmplateAverage) / Tmpstandarddeviation) * ((temp.at<uchar>(k, h) - SrcAverage) / Srcstandarddeviation);
					TmpSquare++;
				}
			}
			NCC[i][j] = sum / TmpSquare;
			sum = 0;
		}
	}

	info.value = NCC[0][0];
	for (int i = 0; i < srcrows; i++) {
		for (int j = 0; j < srccols; j++) {
			if (NCC[i][j] > info.value) {
				info.value = NCC[i][j];
				info.x = i;
				info.y = j;
			}
		}
	}

	for (int i = 0; i < srcrows; i++) {
		delete[] NCC[i];
	}
	delete[] NCC;

	return info;
}



int main() {
	// 读取目标图像和模板
	Mat srcimg = imread("image/lena.jpg", IMREAD_GRAYSCALE);
	if (srcimg.empty()) {
		cout << "could not load image..." << endl;
		return -1;
	}
	// imshow("SourceImg", srcimg);
	Mat templateimg = imread("image/lena2.jpg", IMREAD_GRAYSCALE);
	if (templateimg.empty()) {
		cout << "could not load image..." << endl;
		return -1;
	}
	// imshow("TemplateImage", templateimg);

	// 求图片金字塔
	int levelnum = 4;
	Mat srcimg_p = ImagePyramid(srcimg, levelnum);
	Mat templateimg_p = ImagePyramid(templateimg, levelnum);
	// imshow("SourceImg Pyramid", srcimg_p);
	// imshow("TemplateImage Pyramid", templateimg_p);

	NCCinfo temp, info;
	Mat rotatedimg;
	int anglestep = 90;

	for (int i = 0; i < 360 / anglestep; i++) {
		RotateArbitraryAngle(templateimg_p, rotatedimg, i * anglestep);
		temp = NormalizedCrossCorrelation(srcimg_p, rotatedimg);
		if (info.value < temp.value) {
			info.x = temp.x;
			info.y = temp.y;
			info.value = temp.value;
			info.theta = i * anglestep;
		}
	}

	cout << "The template's x is:" << info.x << endl;
	cout << "The template's y is:" << info.y << endl;
	cout << "The template's theta is:" << info.theta << endl;
	
	// 将检测的区域在图像中表示出来
	Rect rect(info.y * pow(2, levelnum - 1), info.x * pow(2, levelnum - 1), templateimg.cols, templateimg.rows);
	rectangle(srcimg, rect, Scalar(255), 1, LINE_8, 0);
	imshow("Processed_Img", srcimg);

	cout << "The program has been finished!" << endl;
	// 等待任意按键按下
	waitKey(0);
	return 0;
}