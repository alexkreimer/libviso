#ifndef _MVG_H_
#define _MVG_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

#include "misc.h"

using namespace std;
using cv::Mat;
using cv::Mat_;
using cv::Vec4f;

using Eigen::MatrixXd;

template<class T>
Mat slice(const Mat& x, const vector<int>& rows, const vector<int> &cols)
{
    Mat res(rows.size(), cols.size(), x.type());
    vector<int>::const_iterator row_it = rows.begin(), col_it;
    for (int i=0; row_it != rows.end(); ++row_it, ++i)
    {
        int j;
        for (j=0, col_it=cols.begin(); col_it != cols.end(); ++col_it, ++j)
        {
            res.at<T>(i, j) = x.at<T>(*row_it, *col_it);
        }
    }
    return res;
}

vector<int> arange(int range);

/*** Compute fundamental matrix from two camera matrices: x2'*F*x1 = 0
     Overall scale of F is unique and such that, for any X, P1, P2, it is
     F*x1 = vgg_contreps(e2)*x2, where
     x1 = P1*X, x2 = P2*X, e2 = P2*C1, C1 = vgg_wedge(P1).
     Hartley, Zisserman 2nd Ed. pp412
*/
template<class T>
Mat
F_from_P(Mat P1, Mat P2)
{
    // Xj is P1 with row j omitted; Yj is P2 with row j omitted
    Mat X0 = slice<T>(P1, vector<int>({1, 2}), arange(P1.cols));
    Mat X1 = slice<T>(P1, vector<int>({2, 0}), arange(P1.cols));
    Mat X2 = slice<T>(P1, vector<int>({0, 1}), arange(P1.cols));
    Mat Y0 = slice<T>(P2, vector<int>({1, 2}), arange(P1.cols));
    Mat Y1 = slice<T>(P2, vector<int>({2, 0}), arange(P1.cols));
    Mat Y2 = slice<T>(P2, vector<int>({0, 1}), arange(P1.cols));
    Mat M1 = vcat<T>(X0,Y0);
    Mat M2 = vcat<T>(X1,Y0);
    Mat M3 = vcat<T>(X2,Y0);
    Mat M4 = vcat<T>(X0,Y1);
    Mat M5 = vcat<T>(X1,Y1);
    Mat M6 = vcat<T>(X2,Y1);
    Mat M7 = vcat<T>(X0,Y2);
    Mat M8 = vcat<T>(X1,Y2);
    Mat M9 = vcat<T>(X2,Y2);
    Mat F = (Mat_<T>(3,3) <<
	     determinant(M1), determinant(M2), determinant(M3),
	     determinant(M4), determinant(M5), determinant(M6),
	     determinant(M7), determinant(M8), determinant(M9));
    return F;
}



//MatrixXd F_from_P(Eigen::MatrixXd p1, MatrixXd p2);

/*** P = K*[R t] */
template<class T>
Mat P_from_KRt(const Mat &K, const Mat &R, const Mat &t);

/* linear triangulation */
Mat
triangulate_dlt(const Mat &x1, const Mat &x2, const Mat &P1,const Mat &P2);

Mat
triangulate_rectified(const Mat& x1, /* pixel coordinates in the 1st image */
                      const Mat& x2, /* pixel coordinates in the 2nd image */
                      double f, /* focal distance*/
                      double base, /* camera base line distance*/
                      double c1u, /*center of projections in the 1st image */
                      double c1v /* center of projection in the 2nd image */);

/* central projection */
class Camera {
public:
    // intrinics
    Mat K;
    // distortion params
    Vec4f D;
};

class StereoCam {
public:
    Camera c1,c2;
    /* rotation c1 -> c2 */
    Mat R, t;
    Mat p1() const {
	return P_from_KRt<float>(c1.K, Mat::eye(3, 3, CV_32FC1), Mat::zeros(1, 3, CV_32FC1));
    }
    Mat p2() const {
 	return P_from_KRt<float>(c2.K, R, t);
    }
    Mat F() const {
	return F_from_P<float>(p1(),p2());
    }

    /* rectification information */
    /* rotation of c1/c2 to get a rectified pair */
    Mat R1, R2;
    /* rectified camera matrices */
    Mat P1, P2;
    Mat Q;
};


#endif // _MVG_H_
