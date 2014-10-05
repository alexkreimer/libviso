#include "mvg.h"
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>

#include "viso.h"

using cv::Mat_;
using cv::SVD;
using cv::DataType;

template<class T>
Mat rms(const Mat &X1, const Mat &X2)
{
    Mat X = X1-X2;
    X = X.mul(X);
    Mat res(1, X.cols, X.type(), Scalar(0));
    for(int i=0; i<X.rows; ++i)
    {
        for(int j=0; j<X.cols; ++j)
        {
            res.at<T>(0, j) += X.at<T>(i,j);
        }
    }
    for(int j=0; j<X.cols; ++j)
    {
        res.at<T>(0, j) = sqrt(res.at<T>(0,j));
    }
    return res;
}


vector<int> arange(int range)
{
    assert (range>=0);

    vector<int> v(range);
    for(int i=0; i<range; ++i)
        v.at(i) = i;

    return v;
}



#if 0
MatrixXd
F_from_P(MatrixXd P1, MatrixXd P2)
{
    // Xj is P1 with row j omitted; Yj is P2 with row j omitted
    MatrixXd X0(2,4) << P1.row(1), P1.row(2);
    MatrixXd X1(2,4) << P1.row(2), P1.row(0);
    MatrixXd X2(2,4) << P1.row(0), P1.row(1);
    MatrixXd Y0(2,4) << P2.row(1), P2.row(2);
    MatrixXd Y1(2,4) << P2.row(2), P2.row(0);
    MatrixXd Y2(2,4) << P2.row(0), P2.row(1);
    MatrixXd M1(4,4) << X0, Y0;
    MatrixXd M2(4,4) << X1, Y0;
    MatrixXd M3(4,4) << X2, Y0;
    MatrixXd M4(4,4) << X0, Y1;
    MatrixXd M5(4,4) << X1, Y1;
    MatrixXd M6(4,4) << X2, Y1;
    MatrixXd M7(4,4) << X0, Y2;
    MatrixXd M8(4,4) << X1, Y2;
    MatrixXd M9(4,4) << X2, Y2;
    MatrixXd F << M1.determinant(), M2.determinant(), M3.determinant(),
	     M4.determinant(), M5.determinant(), M6.determinant(),
	     M7.determinant(), M8.determinant(), M9.determinant();
    return F;
}
#endif

void test_F_from_P()
{
    Mat P1 =  (Mat_<float>(3,4) << 
               1,0,0,0,
               0,1,0,0,
               0,0,1,0);
    Mat P2 = (Mat_<float>(3,4) <<
              1,0,0,1,
              0,1,0,0,
              0,0,1,0);
    Mat F = F_from_P<float>(P1, P2);
    Mat F_true = (Mat_<float>(3,3) <<
              0,0,0,
              0,0,1,
              0,-1,0);
    assert(norm(F-F_true) == 0);
}

/*** P = K*[R t] */
template<class T>
Mat P_from_KRt(const Mat &K, const Mat &R, const Mat &t)
{
    Mat P(3, 4, CV_32FC1);
    for(int i=0; i<3; ++i)
    {
	for(int j=0; j<3; ++j)
        {
	    P.at<float>(i, j) = R.at<float>(i, j);
	}
    }
    P.at<float>(0,3) = t.at<T>(0);
    P.at<float>(1,3) = t.at<T>(1);
    P.at<float>(2,3) = t.at<T>(2);
    return K*P;
}


/** DLT algorithm for triangulation (LSE).  Based on MVG 2nd Ed, Hartley, Zisserman, pp 312
    
    We search for 3d points X s.t. x1[:,i] = P1*X[:,i] and x2[:,i] = P2*X[:,i]

    DLT uses the fact that to say x1[:,i] = P1*X[:,i] is almost the same as to say \cross(x1[:,i],P1*X[:,i])=0
    The above cross product may be broken in 3 linear equations:
    xp.row(3) − (p 1 T X ) = 0
    y(p 3 T X ) − (p 2 T X ) = 0
    x(p 2 T X ) − y(p 1 T X ) = 0
    @param x1 inhomogenious image points in 1st image x1 \in R^{2,m}, m is the number of points
    @param P1 1st camera matrix
    @param x2 inhomogenious image points in 2nd image; x2 \in R^{2,m}, m is the number of points
    @param P2 2nd camera matrix
*/
Mat
triangulate_dlt(const Mat &x1, const Mat &x2, const Mat &P1,const Mat &P2)
{
    assert(x1.cols == x2.cols);
    assert(x1.rows == 2);
    assert(x2.rows == 2);
    assert(x1.type() == cv::DataType<float>::type);
    assert(x2.type() == cv::DataType<float>::type);
    assert(P1.type() == cv::DataType<double>::type);
    assert(P2.type() == cv::DataType<double>::type);
    int nps = x1.cols;
    Mat X(3, nps, cv::DataType<float>::type);
    for (int i=0; i<nps; ++i) 
    {
        Mat A = Mat::zeros(4, 4, CV_64F);
	//A.row(0) = x1.at<float>(0, i)*P1.row(2)-P1.row(0);
	A.at<double>(0,0) = x1.at<float>(0,i)*P1.at<double>(2,0)-P1.at<double>(0,0);
	A.at<double>(0,1) = x1.at<float>(0,i)*P1.at<double>(2,1)-P1.at<double>(0,1);
	A.at<double>(0,2) = x1.at<float>(0,i)*P1.at<double>(2,2)-P1.at<double>(0,2);
        A.at<double>(0,3) = x1.at<float>(0,i)*P1.at<double>(2,3)-P1.at<double>(0,3);
	//A.row(1) = x1.at<float>(1, i)*P1.row(2)-P1.row(1);
	A.at<double>(1,0) = x1.at<float>(1,i)*P1.at<double>(2,0)-P1.at<double>(1,0);
	A.at<double>(1,1) = x1.at<float>(1,i)*P1.at<double>(2,1)-P1.at<double>(1,1);
	A.at<double>(1,2) = x1.at<float>(1,i)*P1.at<double>(2,2)-P1.at<double>(1,2);
        A.at<double>(1,3) = x1.at<float>(1,i)*P1.at<double>(2,3)-P1.at<double>(1,3);

	// A.row(2) = x2.at<float>(0, i)*P2.row(2)-P2.row(0);
	A.at<double>(2,0) = x2.at<float>(0,i)*P2.at<double>(2,0)-P2.at<double>(0,0);
	A.at<double>(2,1) = x2.at<float>(0,i)*P2.at<double>(2,1)-P2.at<double>(0,1);
	A.at<double>(2,2) = x2.at<float>(0,i)*P2.at<double>(2,2)-P2.at<double>(0,2);
        A.at<double>(2,3) = x2.at<float>(0,i)*P2.at<double>(2,3)-P2.at<double>(0,3);

	//row(3) = x2.at<float>(1, i)*P2.row(2)-P2.row(1);
	A.at<double>(3,0) = x2.at<float>(1,i)*P2.at<double>(2,0)-P2.at<double>(1,0);
	A.at<double>(3,1) = x2.at<float>(1,i)*P2.at<double>(2,1)-P2.at<double>(1,1);
	A.at<double>(3,2) = x2.at<float>(1,i)*P2.at<double>(2,2)-P2.at<double>(1,2);
        A.at<double>(3,3) = x2.at<float>(1,i)*P2.at<double>(2,3)-P2.at<double>(1,3);

        SVD svd = SVD(A, cv::SVD::MODIFY_A);
        double d = (fabs(svd.vt.at<double>(3,3)) < DBL_MIN) ? 1.0 : svd.vt.at<double>(3,3);
	X.at<float>(0,i) = (float)svd.vt.at<double>(3,0)/d;
	X.at<float>(1,i) = (float)svd.vt.at<double>(3,1)/d;
	X.at<float>(2,i) = (float)svd.vt.at<double>(3,2)/d;
    }
    return X;
}


Mat
triangulate_rectified(const Mat& x1, /* pixel coordinates in the 1st image */
                      const Mat& x2, /* pixel coordinates in the 2nd image */
                      double f, /* focal distance*/
                      double base, /* camera base line distance*/
                      double c1u, /*center of projections in the 1st image */
                      double c1v /* center of projection in the 2nd image */)
{
    assert(x1.cols == x2.cols);
    assert(x1.type() == DataType<float>::type);
    assert(x2.type() == DataType<float>::type);
    Mat X(3, x1.cols, DataType<float>::type);
    for (int i=0; i<X.cols; ++i)
    {
        double d = max(x1.at<float>(0,i)-x2.at<float>(0,i),0.0001f);
        X.at<float>(0,i) = (x1.at<float>(0,i)-c1u)*base/d;
        X.at<float>(1,i) = (x1.at<float>(1,i)-c1v)*base/d;
        X.at<float>(2,i) = f*base/d;
    }
    return X;
}
