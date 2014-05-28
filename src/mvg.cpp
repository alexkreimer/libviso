#include "mvg.h"
#include <iostream>


/*** Compute fundamental matrix from two camera matrices: x2'*F*x1 = 0
     Overall scale of F is unique and such that, for any X, P1, P2, it is
     F*x1 = vgg_contreps(e2)*x2, where
     x1 = P1*X, x2 = P2*X, e2 = P2*C1, C1 = vgg_wedge(P1).
     Hartley, Zisserman 2nd Ed. pp412
*/

Mat
F_from_P(Mat P1, Mat P2)
{
    Mat X1(0, 4, CV_32FC1), X2, X3, Y1, Y2, Y3; // no push_back for Matx
    Mat M1, M2, M3, M4, M5, M6, M7, M8, M9;
    X1.push_back(P1.row(1));
    X1.push_back(P1.row(2));
    X2.push_back(P1.row(2));
    X2.push_back(P1.row(0));
    X3.push_back(P1.row(0));
    X3.push_back(P1.row(1));
    Y1.push_back(P2.row(1));
    Y1.push_back(P2.row(2));
    Y2.push_back(P2.row(2));
    Y2.push_back(P2.row(0));
    Y3.push_back(P2.row(0));
    Y3.push_back(P2.row(1));
    vconcat(X1,Y1,M1);
    vconcat(X2,Y1,M2);
    vconcat(X3,Y1,M3);
    vconcat(X1,Y2,M4);
    vconcat(X2,Y2,M5);
    vconcat(X3,Y2,M6);
    vconcat(X1,Y3,M7);
    vconcat(X2,Y3,M8);
    vconcat(X3,Y3,M9);
    Mat F = (Mat_<float>(3,3) <<
	     determinant(M1), determinant(M2), determinant(M3),
	     determinant(M4), determinant(M5), determinant(M6),
	     determinant(M7), determinant(M8), determinant(M9));
    return F;
}

/*** P = K*[R t] */
Mat
P_from_KRt(const Mat &K,
	   const Mat &R,
	   const Vec3f &t)
{
    Mat P(3, 4, CV_32FC1);
    for(int i=0; i<3; ++i) {
	for(int j=0; j<3; ++j) {
	    P.at<float>(i,j) = R.at<float>(i,j);
	}
    }
    P.at<float>(0,3) = t[0];
    P.at<float>(1,3) = t[1];
    P.at<float>(2,3) = t[2];
    P.at<float>(3,3) = 1.0f;
    return K*P;
}

