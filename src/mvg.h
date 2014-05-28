#ifndef _MVG_H
#define _MVG_H

#include<opencv2/core/core.hpp>

using namespace cv;

/*** Compute fundamental matrix from two camera matrices: x2'*F*x1 = 0
     Overall scale of F is unique and such that, for any X, P1, P2, it is
     F*x1 = vgg_contreps(e2)*x2, where
     x1 = P1*X, x2 = P2*X, e2 = P2*C1, C1 = vgg_wedge(P1).
     Hartley, Zisserman 2nd Ed. pp412
*/
Mat
F_from_P(Mat P1, Mat P2);

/*** P = K*[R t] */
Mat
P_from_KRt(const Mat &K,
	   const Mat &R,
	   const Vec3f &t);
#endif
