#include "Rectify.h"

#include <algorithm>
#include <fstream>

using namespace cv;
using namespace std;

#define GRAY(_R_, _G_, _B_) (((_R_) * 0.27) + ((_G_) * 0.67) + ((_B_) * 0.06))

#define LOOP_MAT(__mat__) for(unsigned int row=0;row<(__mat__).rows;row++)\
                                for(unsigned int col=0;col<(__mat__).cols;col++)

#define LOOP_BOUNDARY(__bound__) for(unsigned int row=(__bound__).B;row<=(__bound__).T;row++)\
                                for(unsigned int col=(__bound__).L;col<=(__bound__).R;col++) 

#define IMG_BOUNDARY(__mat__, __bound__) (__mat__).colRange((__bound__).L, (__bound__).R + 1).rowRange((__bound__).B, (__bound__).T + 1)

#define INSIDE(__mat__, __row__, __col__) ((((__row__) >= 0) && ((__row__) < __mat__.rows)) && (((__col__) >= 0) && ((__col__) < __mat__.cols)))

cv::Mat RectPano::RectWarpping(cv::Mat& input)
{
	cv::Mat img = Resize(input);       // return resized image (fit to bounding box)
	cv::Mat dm = LocalWarpping(img); // return displacement map

	cout << "Mesh Backward Warpping....";
	Grid meshGrid(img, 20, 20);
	MeshBackwardWarpping(dm, meshGrid);
	cout << "\rMesh Backward Warpping...Done." << endl;

	Mat result = GlobalWarpping(img, meshGrid);
	return result;
}

cv::Mat RectPano::Resize(cv::Mat& img)
{
	Boundary boundingBox;
	boundingBox.T = 0;
	boundingBox.B = INT_MAX;
	boundingBox.R = 0;
	boundingBox.L = INT_MAX;

	LOOP_MAT(img) {
		Vec4b& color = img.at<Vec4b>(row, col);
		if (color[3] != 0) {
			boundingBox.T = max(row, boundingBox.T);
			boundingBox.B = min(row, boundingBox.B);

			boundingBox.R = max(col, boundingBox.R);
			boundingBox.L = min(col, boundingBox.L);
		}
	}

	Mat result = IMG_BOUNDARY(img, boundingBox);

	return result;
}

cv::Mat RectPano::LocalWarpping(cv::Mat& img)
{
	Mat res = img.clone(); // for seam carving
	Mat dm(res.rows, res.cols, CV_32SC2); // displacement map

	cout << endl << "Local Warpping....";
	for (int iter = 0; iter < 999999; iter++) {
		Boundary bound = FindBoundarySegment(res);
		int area = bound.area();
		if (area > 1) {
			Mat grad = Gradient(res); // img XY gradient
			LocalSeamCarving(res, grad, dm, bound);
		}
		else {
			break;
		}
	}

	// fill missing pixels with nearest neighbor
	LOOP_MAT(res) {
		Vec4b& color = res.at<Vec4b>(row, col);
		Vec2i& disp  = dm.at<Vec2i>(row, col);
		if (color[3] == 0) {
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {

					int krow = row + i;
					int kcol = col + j;

					if (!INSIDE(res, krow, kcol)) {
						continue;
					}

					Vec4b& kcolor = res.at<Vec4b>(krow, kcol);
					if (kcolor[3] > 0) {
						Vec2i& kdisp = dm.at<Vec2i>(krow, kcol);
						color = kcolor;
						disp = kdisp;
						i = j = 2; // break
						break;
					}
				}
			}
		}
	}
	cout << "\rLocal Warpping...Done."<< endl;

	/*LOOP_MAT(dm) { // visaulize displacment map
		Vec2i& disp = dm.at<Vec2i>(row, col);
		res.at<Vec4b>(row, col) = Vec4b(abs(disp[0]) + abs(disp[1]),0,0,255);
	}*/

	return dm;
}

RectPano::Boundary RectPano::FindBoundarySegment(cv::Mat& img)
{
	Boundary maxBound;
	int maxLength = 0;

	// left
	FindBoundarySide(img, 0, img.rows, 0, 1, maxLength, maxBound);
	// right
	FindBoundarySide(img, 0, img.rows, img.cols - 1, img.cols, maxLength, maxBound);
	// top
	FindBoundarySide(img, 0, 1, 0, img.cols, maxLength, maxBound);
	// bottom
	FindBoundarySide(img, img.rows - 1, img.rows, 0, img.cols, maxLength, maxBound);

	return maxBound;
}

void RectPano::FindBoundarySide(cv::Mat& img, int srow, int erow, int scol, int ecol, int& maxLength, RectPano::Boundary& maxBound)
{
	bool isRowSides = ((erow - srow) > (ecol - scol));

	int lrow = srow, lcol = scol; // last row & col
	int crow = srow, ccol = scol; // current row & col

	int length = 0;
	for (int row = srow; row < erow; row++) {
		for (int col = scol; col < ecol; col++) {

			Vec4b& color = img.at<Vec4b>(row, col);

			// if alpha is empty, continue counting
			if (color[3] == 0) {
				length += 1;
				if (isRowSides) {
					crow = row;
				}
				else {
					ccol = col;
				}
			}
			else { // stop counting and calculate distance

				if (length > maxLength) {
					maxLength = length;
					maxBound.B = lrow;
					maxBound.T = crow;
					maxBound.L = lcol;
					maxBound.R = ccol;
				}

				length = 0;

				if (isRowSides) {
					lrow = row + 1;
				}
				else {
					lcol = col + 1;
				}
			}

		}
	}

	if (length > maxLength) {
		maxLength = length;
		maxBound.B = lrow;
		maxBound.T = crow;
		maxBound.L = lcol;
		maxBound.R = ccol;
	}
}

cv::Mat RectPano::Gradient(cv::Mat & img)
{
	Mat grad(img.rows, img.cols, CV_32FC2);
	int w[3][3] = { 
		{1, 0,-1},
		{2, 0,-2},
		{1, 0,-1},
	};

	// calculate gradient
	LOOP_MAT(img) {

		Vec2f& gI = grad.at<Vec2f>(row, col);
		gI[0] = 0;
		gI[1] = 0;

		Vec4b& color = img.at<Vec4b>(row, col);

		if (color[3] == 0) {
			gI[0] = INT_MAX;
			gI[1] = INT_MAX;
		}
		else {
			// sobel kernel
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {

					int krow = row + i;
					int kcol = col + j;

					if (!INSIDE(img, krow, kcol)) {
						continue;
					}

					Vec4b& kcolor = img.at<Vec4b>(krow, kcol);

					int w_x = w[i + 1][j + 1];
					int w_y = w[j + 1][i + 1];

					gI[0] += (w_x * GRAY(kcolor[0], kcolor[1], kcolor[2]));
					gI[1] += (w_y * GRAY(kcolor[0], kcolor[1], kcolor[2]));
				}
			}
		}
		
		gI[0] = abs(gI[0]);
		gI[1] = abs(gI[1]);
	}

	return grad;
}

void RectPano::LocalSeamCarving(cv::Mat & img, cv::Mat & grad, cv::Mat & dm, Boundary& bound)
{
	bool isRowSides = (bound.L == bound.R);

	vector<Vec2i> seam; // store (col, row)

	// accumulated gradient for finding min difference seam.
	Mat gradSum(img.rows, img.cols, CV_32FC2);
	if (isRowSides) {
		// init gradSum
		for (int row = bound.B; row <= bound.T; row++) {
			for (int col = 0; col < img.cols; col++) {
				gradSum.at<Vec2f>(row, col) = Vec2f(grad.at<Vec2f>(row, col)[1], 0);
			}
		}

		// calc gradSum accumulate from bottom to top
		for (int row = bound.B + 1; row <= bound.T; row++) {
			for (int col = 0; col < img.cols; col++) {

				Vec2f& Sum_idx = gradSum.at<Vec2f>(row, col);
				float minIy = INT_MAX;
				for (int i = -1; i <= 1; i++) {

					int krow = row - 1;
					int kcol = col + i;

					if (INSIDE(gradSum, krow, kcol)) {
						float Iy = gradSum.at<Vec2f>(krow, kcol)[1] + gradSum.at<Vec2f>(krow, kcol)[0];
						if (Iy < minIy) { // compare min y gradient
							minIy = Iy;
							Sum_idx[1] = i;
						}
					}
				}

				// accumulate gradient value
				if (minIy == INT_MAX) {
					Sum_idx[0] = INT_MAX;
				}
				else {
					Sum_idx[0] += minIy;
				}
			}
		}

		// find min seam
		float minSum = INT_MAX;
		int seamCol   = 0;
		for (int col = 0; col < img.cols; col++) {
			Vec2f& Sum_idx = gradSum.at<Vec2f>(bound.T, col);
			if (Sum_idx[0] < minSum) {
				minSum = Sum_idx[0];
				seamCol = col;
			}
		}

		for (int row = bound.T; row > bound.B; row--) {
			Vec2f& Sum_idx = gradSum.at<Vec2f>(row, seamCol);
			seam.push_back(Vec2i(seamCol, row));
			seamCol += Sum_idx[1];
		}
	}
	else { // handle colSides

		// init gradSum with Ix
		for (int col = bound.L; col <= bound.R; col++) {
			for (int row = 0; row < img.rows; row++) {
				gradSum.at<Vec2f>(row, col) = Vec2f(grad.at<Vec2f>(row, col)[0], 0);
			}
		}

		// calc gradSum accumulate from bottom to top
		for (int col = bound.L + 1; col <= bound.R; col++) {
			for (int row = 0; row < img.rows; row++) {

				Vec2f& Sum_idx = gradSum.at<Vec2f>(row, col);
				float minIx = INT_MAX;

				for (int i = -1; i <= 1; i++) {

					int krow = row + i;
					int kcol = col - 1;

					if (INSIDE(gradSum, krow, kcol)) {
						float Ix = gradSum.at<Vec2f>(krow, kcol)[0] + gradSum.at<Vec2f>(krow, kcol)[1];
						if (Ix < minIx) { // compare min y gradient
							minIx = Ix;
							Sum_idx[1] = i;
						}
					}
				}

				// accumulate gradient value
				if (minIx == INT_MAX) {
					Sum_idx[0] = INT_MAX;
				}
				else {
					Sum_idx[0] += minIx;
				}
			}
		}

		// find min seam
		float minSum = INT_MAX;
		int seamRow = 0;
		for (int row = 0; row < img.rows; row++) {
			Vec2f& Sum_idx = gradSum.at<Vec2f>(row, bound.R);
			if (Sum_idx[0] < minSum) {
				minSum = Sum_idx[0];
				seamRow = row;
			}
		}

		for (int col = bound.R; col > bound.L; col--) {
			Vec2f& Sum_idx = gradSum.at<Vec2f>(seamRow, col);
			seam.push_back(Vec2i(col, seamRow));
			seamRow += Sum_idx[1];
		}
	}

	// apply seam
	if (isRowSides) {
		bool isShiftLeft = (bound.L == 0);
		// copy & shift seam pixel
		// copy & add offset to displacements

		for (int i = 0; i < seam.size(); i++) {
			Vec2i& p = seam[i];

			if (isShiftLeft) {
				Mat sub_img0 = img.rowRange(p[1], p[1] + 1).colRange(0, p[0]);
				Mat sub_img1 = img.rowRange(p[1], p[1] + 1).colRange(1, p[0] + 1).clone();
				sub_img1.copyTo(sub_img0);

				Mat sub_dm0 = dm.rowRange(p[1], p[1] + 1).colRange(0, p[0]);
				Mat sub_dm1 = dm.rowRange(p[1], p[1] + 1).colRange(1, p[0] + 1).clone();
				sub_dm1.copyTo(sub_dm0);

				for (int col = 0; col < p[0]; col++) {
					dm.at<cv::Vec2i>(p[1], col)[0] -= 1;
				}
			}
			else {
				Mat sub_img0 = img.rowRange(p[1], p[1] + 1).colRange(p[0] + 1, img.cols);
				Mat sub_img1 = img.rowRange(p[1], p[1] + 1).colRange(p[0], img.cols - 1).clone();
				sub_img1.copyTo(sub_img0);

				Mat sub_dm0 = dm.rowRange(p[1], p[1] + 1).colRange(p[0] + 1, img.cols);
				Mat sub_dm1 = dm.rowRange(p[1], p[1] + 1).colRange(p[0], img.cols - 1).clone();
				sub_dm1.copyTo(sub_dm1);

				for (int col = p[0]+1; col < img.cols; col++) {
					dm.at<cv::Vec2i>(p[1], col)[0] += 1;
				}
			}

		}
	}
	else {
		bool isShiftBottom = (bound.B == 0);
		for (int i = 0; i < seam.size(); i++) {
			Vec2i& p = seam[i];
			if (isShiftBottom) {
				Mat sub_img0 = img.colRange(p[0], p[0] + 1).rowRange(0, p[1]);
				Mat sub_img1 = img.colRange(p[0], p[0] + 1).rowRange(1, p[1] + 1).clone();
				sub_img1.copyTo(sub_img0);

				Mat sub_dm0 = dm.colRange(p[0], p[0] + 1).rowRange(0, p[1]);
				Mat sub_dm1 = dm.colRange(p[0], p[0] + 1).rowRange(1, p[1] + 1).clone();
				sub_dm1.copyTo(sub_dm0);

				for (int row = 0; row < p[1]; row++) {
					dm.at<cv::Vec2i>(row, p[0])[1] -= 1;
				}
			}
			else {
				Mat sub_img0 = img.colRange(p[0], p[0] + 1).rowRange(p[1] + 1, img.rows);
				Mat sub_img1 = img.colRange(p[0], p[0] + 1).rowRange(p[1], img.rows - 1).clone();
				sub_img1.copyTo(sub_img0);

				Mat sub_dm0 = dm.colRange(p[0], p[0] + 1).rowRange(p[1] + 1, img.rows);
				Mat sub_dm1 = dm.colRange(p[0], p[0] + 1).rowRange(p[1], img.rows - 1).clone();
				sub_dm1.copyTo(sub_dm0);

				for (int row = p[1]+1; row < img.rows; row++) {
					dm.at<cv::Vec2i>(row, p[0])[1] += 1;
				}
			}
		}
	}
}

void RectPano::MeshBackwardWarpping(Mat & dm, Grid & meshGrid)
{
	cv::Mat patch;
	for (Point2f& v : meshGrid.vertices) {
		Vec2i& disp = dm.at<Vec2i>(v.y, v.x);
		//cout << disp[0] << ", " << disp[1] << endl;
		v.x -= disp[0];
		v.y -= disp[1];
	}
}


bool intersection(Point2f& o1, Point2f& o2, Point2f& p1, Point2f& p2, Point2f &intersection)
{
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < 0.001)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	intersection = o1 + d1 * t1;

	return true;
}
cv::Mat RectPano::GlobalWarpping(cv::Mat & img, Grid & meshGrid)
{
	cout << "Global Warpping...." << endl;

	cout << "\t Detecting Line Segments....";
	vector<Grid::Line> lineSegments;
	LineDetect(img, meshGrid, lineSegments);
	cout << "\r\t Detecting Line Segments...Done." << endl;

	return ShowMeshGrid(img, meshGrid);

	Grid warppedGrid(meshGrid);
	int iterations = 1;
	for (int iter = 0; iter < 1; iter++) {
		cout << "\t Solving Energy Function..." << ((iter * 3) / (iterations * 3.0f)) << "                   \r";
		SolveShapeAndBound(img, warppedGrid);
		cout << "\t Solving Energy Function..." << ((iter * 3 + 1) / (iterations * 3.0f)) << "                   \r";
		//SolveLine(meshGrid);
		cout << "\t Solving Energy Function..." << ((iter * 3 + 2) / (iterations * 3.0f)) << "                   \r";
	}
	cout << "\t Solving Energy Function...Done.                   " << endl;

#pragma region ImageWarpping
	cout << "\t Warpping Image....";
	// Warpping Image from input mesh to solved mesh
	Mat rgbImg;
	cv::cvtColor(img, rgbImg, COLOR_BGRA2BGR);
	Mat result = img.clone();
	LOOP_MAT(rgbImg) {
		Point2f warppedPixel(col, row);

		int qIdx = warppedGrid.FindClosestQuad(warppedPixel);

		if (qIdx != -1) {
			Point2f qUV = warppedGrid.ConvertToQuadSpace(qIdx, warppedPixel);

			Point2f imgPixel = meshGrid.ConvertToImageSpace(qIdx, qUV);

			cv::Mat patch;
			cv::getRectSubPix(rgbImg, cv::Size(1, 1), imgPixel, patch);
			Vec3b& color = patch.at<Vec3b>(0, 0);

			result.at<Vec4b>(row, col) = Vec4b(color[0], color[1], color[2], 255);
		}
		else {
			result.at<Vec4b>(row, col) = Vec4b(255, 0, 0, 255);
		}
	}
	cout << "\r\t Warpping Image...Done." << endl;
#pragma endregion

	cout << "Global Warpping...Done." << endl;

	return ShowMeshGrid(result, warppedGrid);
}

using namespace ximgproc;
void RectPano::LineDetect(cv::Mat & img, Grid & meshGrid, std::vector<Grid::Line>& lineSegs)
{
	int    length_threshold = 10;
	float  distance_threshold = 1.41421356f;
	double canny_th1 = 50.0;
	double canny_th2 = 50.0;
	int    canny_aperture_size = 3;
	bool   do_merge = false;

	Ptr<FastLineDetector> FLD = createFastLineDetector(
		length_threshold,
		distance_threshold,
		canny_th1,
		canny_th2,
		canny_aperture_size,
		do_merge);

	vector<Vec4f> lines;
	cv::Mat gray; // only process grayscale image
	cv::cvtColor(img, gray, COLOR_BGRA2GRAY);
	FLD->detect(gray, lines);

	// keep lines inside border
#pragma region RemoveOutliner
	vector<Grid::Line> lines_inside;
	int testRange = 5;
	int testSamples[4][2] = {
		{-testRange,-testRange},
		{-testRange, testRange},
		{testRange, testRange},
		{testRange, -testRange},
	};
	for (int i = 0; i < lines.size(); i++) {
		bool outside = true;

		for (int j = 0; j < 2; j++) {
			Point2f p(lines[i][j*2], lines[i][j*2+1]);

			bool sampleInside = true;
			for (int k = 0; k < 4; k++) {
				int krow = p.y + testSamples[k][1];
				int kcol = p.x + testSamples[k][0];

				if (INSIDE(img, krow, kcol)) {
					Vec4b& color = img.at<Vec4b>(krow, kcol);
					if (color[3] == 0) {
						sampleInside = false;
						break;
					}
				}
				else {
					sampleInside = false;
					break;
				}
			}

			if (sampleInside) {
				outside = false;
				break;
			}
		}

		if (!outside) {
			Grid::Line l;
			l.points[0] = Point2f(lines[i][0], lines[i][1]);
			l.points[1] = Point2f(lines[i][2], lines[i][3]);
			lines_inside.push_back(l);
		}
	}
#pragma endregion

	// break lines into line segments intersecting with grid quads
#pragma region Line Intersection
	for (int i = 0; i < lines_inside.size(); i++) {

		Grid::Line& l = lines_inside[i];
		bool intersected = false;
		Point2f intersectPoint;

		// loop all 
		for (int j = 0; j < meshGrid.quads.size(); j++) {
			Point2f& v0 = meshGrid.vertices[meshGrid.quads[j].verts[0]];
			Point2f& v1 = meshGrid.vertices[meshGrid.quads[j].verts[1]];
			Point2f& v2 = meshGrid.vertices[meshGrid.quads[j].verts[2]];
			Point2f& v3 = meshGrid.vertices[meshGrid.quads[j].verts[3]];

			if (intersection(l.points[0], l.points[1], v0, v1, intersectPoint)) {
				intersected = true;
			}
		}

		if (!intersected) {
			lineSegs.push_back(l);
		}
	}
#pragma endregion

	for (int i = 0; i < lineSegs.size(); i++) {
		Grid::Line& l = lineSegs[i];
		cv::line(img, l.points[0] , l.points[1], Scalar(0, 0, 255, 255), 1);
	}
}

void RectPano::SolveShapeAndBound(cv::Mat & img, Grid & meshGrid)
{
	// Shape & boundary Energy

	// init matrix
	int Qs = meshGrid.quads.size();
	int Vs = meshGrid.vertices.size();
	int W = meshGrid.width;
	int H = meshGrid.height;

	Mat E = Mat::zeros(Qs * 8 + H * 2 + W * 2, Vs * 2, CV_32F);
	Mat V = Mat::zeros(Vs * 2, 1, CV_32F);
	Mat b = Mat::zeros(Qs * 8 + H * 2 + W * 2, 1, CV_32F);

	// init quad parts
	Mat Aq(8, 4, CV_32F);
	Mat I = Mat::eye(8, 8, CV_32F); // identity matrix
	for (int qIdx = 0; qIdx < Qs; qIdx++) {
		Grid::Quad& quad = meshGrid.quads[qIdx];

		for (int j = 0; j < 4; j++) {
			int vIdx = quad.verts[j];

			Point2f& v = meshGrid.vertices[vIdx];

			Aq.at<float>(j * 2, 0) = v.x;
			Aq.at<float>(j * 2, 1) = -v.y;
			Aq.at<float>(j * 2, 2) = 1;
			Aq.at<float>(j * 2, 3) = 0;

			Aq.at<float>(j * 2 + 1, 0) = v.y;
			Aq.at<float>(j * 2 + 1, 1) = v.x;
			Aq.at<float>(j * 2 + 1, 2) = 0;
			Aq.at<float>(j * 2 + 1, 3) = 1;

		}

		Mat M = Aq * (Aq.t() * Aq).inv() * Aq.t() - I;
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 4; j++) {
				int vIdx = quad.verts[j];

				float x = M.at<float>(i, j * 2);
				float y = M.at<float>(i, j * 2 + 1);

				E.at<float>(qIdx * 8 + i, vIdx * 2)     = x;
				E.at<float>(qIdx * 8 + i, vIdx * 2 + 1) = y;
			}
		}

	}

	// init bounding parts
	int rowOffset = Qs * 8;
	float weightB = 100;
	// left
	for (int i = 0; i < H; i++) {
		int vIdx = i * W;

		E.at<float>(rowOffset + i, vIdx * 2) = weightB;
		b.at<float>(rowOffset + i, 0) = 0;
	}
	rowOffset = rowOffset + H;

	// right
	for (int i = 0; i < H; i++) {
		int vIdx = i * W + W - 1;

		E.at<float>(rowOffset + i, vIdx * 2) = weightB;
		b.at<float>(rowOffset + i, 0) = (img.cols - 1) * weightB;
	}
	rowOffset = rowOffset + H;

	// bottom
	for (int i = 0; i < W; i++) {
		int vIdx = i;

		E.at<float>(rowOffset + i, vIdx * 2 + 1) = weightB;
		b.at<float>(rowOffset + i, 0) = 0;
	}
	rowOffset = rowOffset + W;

	// top
	for (int i = 0; i < W; i++) {
		int vIdx = (H - 1) * W + i;

		E.at<float>(rowOffset + i, vIdx * 2 + 1) = weightB;
		b.at<float>(rowOffset + i, 0) = (img.rows - 1) * weightB;
	}


	// solve least square EV = b
	solve(E, b, V, DECOMP_NORMAL);

	for (int i = 0; i < Vs; i++) {
		float x = V.at<float>(i * 2, 0);
		float y = V.at<float>(i * 2 + 1, 0);

		meshGrid.vertices[i].x = round(x * 1000) * 0.001;
		meshGrid.vertices[i].y = round(y * 1000) * 0.001;
	}
}

void RectPano::SolveLine(Grid & meshGrid)
{
	// Line Energy

}

cv::Mat RectPano::ShowMeshGrid(Mat & img, Grid & meshGrid)
{
	Mat result = img.clone();
	for (auto& quad : meshGrid.quads) {
		cv::line(result, meshGrid.vertices[quad.verts[0]], meshGrid.vertices[quad.verts[1]], Scalar(0, 255, 0, 255));
		cv::line(result, meshGrid.vertices[quad.verts[1]], meshGrid.vertices[quad.verts[2]], Scalar(0, 255, 0, 255));
		cv::line(result, meshGrid.vertices[quad.verts[2]], meshGrid.vertices[quad.verts[3]], Scalar(0, 255, 0, 255));
		cv::line(result, meshGrid.vertices[quad.verts[3]], meshGrid.vertices[quad.verts[0]], Scalar(0, 255, 0, 255));
	}

	return result;
}

RectPano::RectPano(string inputPath)
{
    this->path = inputPath;

	Mat image = imread(inputPath, cv::IMREAD_UNCHANGED);
	if (!image.data)// Check for invalid input
	{
		cout << "Could not open or find the image : " << inputPath << std::endl;
		throw - 1;
	}

	if (image.type() != CV_8UC4) {
		cout << "Image must be png type!" << std::endl;
		throw - 1;
	}

	this->image = image.clone();
}

RectPano::~RectPano()
{
}

const cv::Mat RectPano::GetImage()
{
    return this->image;
}

const cv::Mat RectPano::GetRectImage()
{
	if (!this->rectImage.data) {
		this->rectImage = RectWarpping(this->image);
	}
	return this->rectImage;
}

RectPano::Boundary::Boundary(unsigned int i)
{
	T = B = L = R = i;
}

int RectPano::Boundary::area()
{
	return (T - B + 1) * (R - L + 1);
}

RectPano::Grid::Grid(Mat& img, int resolutionX, int resolutionY)
{
	width  = resolutionX;
	height = resolutionY;

	vertices.clear();

	float ratioX = (img.cols-1) / (float)(resolutionX-1);
	float ratioY = (img.rows-1) / (float)(resolutionY-1);

	for (int j = 0; j < resolutionY; j++)
	{
		for (int i = 0; i < resolutionX; i++)
		{
			vertices.push_back(Point2f(i * ratioX, j * ratioY));
		}
	}

	// quads
	for (int j = 0; j < resolutionY - 1; j++)
	{
		for (int i = 0; i < resolutionX - 1; i++)
		{
			Quad q;
			q.verts[0] = j * width + i;
			q.verts[1] = j * width + i + 1;
			q.verts[2] = (j + 1) * width + i + 1;
			q.verts[3] = (j + 1) * width + i;
			quads.push_back(q);
		}
	}
}

RectPano::Grid::Grid(Grid & g)
{
	width = g.width;
	height = g.height;

	vertices = g.vertices;
	quads = g.quads;
}

int RectPano::Grid::FindClosestQuad(Point2f p)
{
	float minDistance = INT_MAX;
	int minIdx = -1;

	for (int i = 0; i < quads.size(); i++) {

		bool inside = true;
		for (int j = 0; j < 4; j++) {
			Point2f& v0 = vertices[quads[i].verts[j]];
			Point2f& v1 = vertices[quads[i].verts[(j+1)&3]];

			float cross = (v0 - v1).cross((v0 - p));

			// clockwise
			if (cross < 0) {
				inside = false;
				break;
			}
		}

		if (inside) {
			float distance = 0;
			for (int j = 0; j < 4; j++) {
				Point2f& v = vertices[quads[i].verts[j]];

				distance += (v.x - p.x) * (v.x - p.x) + (v.y - p.y) * (v.y - p.y);
			}

			if (distance < minDistance) {
				minDistance = distance;
				minIdx = i;
			}
		}
	}

	return minIdx;
}

float Cross(Point2f v, Point2f w)
{
	return v.x * w.y - v.y * w.x;
}
Point2f RectPano::Grid::ConvertToQuadSpace(int qIdx, Point2f p)
{
	Point2f& v0 = vertices[quads[qIdx].verts[3]];
	Point2f& v1 = vertices[quads[qIdx].verts[2]];
	Point2f& v2 = vertices[quads[qIdx].verts[0]];
	Point2f& v3 = vertices[quads[qIdx].verts[1]];

	Point2f q =  p - v0;
	Point2f b1 = v1 - v0;
	Point2f b2 = v2 - v0;
	Point2f b3 = v0 - v1 - v2 + v3;

	// Set up quadratic formula
	float A = Cross(b2, b3);
	float B = Cross(b3, q) - Cross(b1, b2);
	float C = Cross(b1, q);

	// Solve for v
	Point2f uv;
	if (abs(A) < 0.01)
	{
		// Linear form
		uv.y = -C / B;
	}
	else
	{
		// Quadratic form. Take positive root for CCW winding with V-up
		float discrim = B*B - 4 * A*C;
		uv.y = 0.5 * (-B + sqrt(discrim)) / A;
	}

	// Solve for u, using largest-magnitude component
	Point2f denom = b1 + uv.y * b3;
	if (abs(denom.x) > abs(denom.y))
		uv.x = (q.x - b2.x * uv.y) / denom.x;
	else
		uv.x = (q.y - b2.y * uv.y) / denom.y;
 
	if (uv.x > 1) {
		uv.x = 1;
	}
	else if (uv.x < 0) {
		uv.x = 0;
	}

	if (uv.y > 1) {
		uv.y = 1;
	}
	else if (uv.y < 0) {
		uv.y = 0;
	}

	return uv;
}

Point2f RectPano::Grid::ConvertToImageSpace(int qIdx, Point2f uv)
{
	Point2f& v0 = vertices[quads[qIdx].verts[3]];
	Point2f& v1 = vertices[quads[qIdx].verts[2]];
	Point2f& v2 = vertices[quads[qIdx].verts[0]];
	Point2f& v3 = vertices[quads[qIdx].verts[1]];

	// p = lerp(lerp(p0,p1,u), lerp(p2,p3,u), v)
	float u = uv.x;
	float v = uv.y;

	return  (1-v) * v0 * (1-u) + (1-v) * v1 * u + v * v2 * (1-u) + v*v3 * u;
}

