#pragma once

#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ximgproc.hpp>
#include <iostream>
#include <string>
#include <vector>

class RectPano {

	class Boundary {
	public:
		Boundary(unsigned int i = 0);
		int T;
		int B;
		int L;
		int R;
		int area();
	};

	class Grid {
	public:
		struct Quad{
			int verts[4];
		};

		struct Line {
			cv::Point2f points[2];
			cv::Point2f uv_points[2];
			int qIdx;
			int mIdx;
			float angle;
		};

		std::vector<cv::Point2f> vertices;
		std::vector<Quad> quads;

		int width;
		int height;

		Grid(cv::Mat& img, int resolutionX, int resolutionY);
		Grid(Grid& g);

		int FindClosestQuad(cv::Point2f p);
		cv::Point2f ConvertToQuadSpace(int qIdx, cv::Point2f p);
		cv::Point2f ConvertToImageSpace(int qIdx, cv::Point2f p);
	};

private:
	std::string path; // 圖片資路徑
	cv::Mat image; // 讀取進來的圖片
	cv::Mat rectImage; // 讀取進來的圖片

	cv::Mat RectWarpping(cv::Mat& img);
	cv::Mat Resize(cv::Mat& img);
	cv::Mat LocalWarpping(cv::Mat& img);
	Boundary FindBoundarySegment(cv::Mat& img);
	void FindBoundarySide(cv::Mat& img, int srow, int erow, int scol, int ecol, int& maxLength, Boundary& maxBound);
	cv::Mat Gradient(cv::Mat& img);
	void LocalSeamCarving(cv::Mat& img, cv::Mat& grad, cv::Mat& dm, Boundary& bound);
	void MeshBackwardWarpping(cv::Mat& dm, Grid& meshGrid);
	cv::Mat GlobalWarpping(cv::Mat& img, Grid& meshGrid);
	void LineDetect(cv::Mat& img, Grid& meshGrid, std::vector<Grid::Line>& lineSegs);
	void LineQuantize(std::vector<Grid::Line>& lineSegs);
	void SolveEnergy(cv::Mat& img, Grid& meshGrid, std::vector<Grid::Line>& lineSegs, float* orientBin);
	void SolveLineTheta(cv::Mat& img, Grid& meshGrid, std::vector<Grid::Line>& lineSegs, float* orientBin);


	cv::Mat ShowMeshGrid(cv::Mat& img, Grid& meshGrid);
public: 

	RectPano(std::string inputPath);
	~RectPano();

	const cv::Mat GetImage();
	const cv::Mat GetRectImage();
};

