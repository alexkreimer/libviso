#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "triangulation.h"

extern LoggerPtr logger;

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
vector<Point3f>
triangulate_dlt(const vector<Point2f>& x1, const vector<Point2f>& x2,
		Mat P1, Mat P2)
{
    LOG4CXX_DEBUG(logger, "dlt triangulation...");
    assert(x1.size() == x2.size());
    int npts = x1.size();
    Mat A = Mat::zeros(4*npts, 3, CV_32F);
    for (int i=0; i<npts; ++i) 
    {
	A.row(i+0) = Mat(x1.at(i).x*P1.row(2)-P1.row(0));
	A.row(i+1) = Mat(x1.at(i).y*P1.row(2)-P1.row(1));
	A.row(i+2) = Mat(x2.at(i).x*P2.row(2)-P2.row(0));
	A.row(i+3) = Mat(x2.at(i).y*P2.row(2)-P2.row(1));
    }
    SVD svd = SVD(A, cv::SVD::MODIFY_A);
    vector<Point3f> X(npts);
    for(int i=0; i<npts; ++i)
    {
	X.at(i).x = svd.vt.at<float>(3*i+0);
	X.at(i).y = svd.vt.at<float>(3*i+1);
	X.at(i).z = svd.vt.at<float>(3*i+2);
    }
    LOG4CXX_DEBUG(logger, "dlt triangulation is done");
    return X;
}

/**From "Triangulation", Hartley, and Sturm, Computer vision and image understanding, 1997.

   @param u homegenious image points in 1st image plane
   @param P 1st camera matrix
   @param u1 homogenious image point in 2nd image plane
   @param P1 2nd camera matrix
*/
#if 0
cv::Mat
LinearLSTriangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1)
{
    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    cv::Matx43d A(u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
		  u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
		  u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
		  u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2));
    cv::Matx41d B(-(u.x*P(2,3)-P(0,3)),
		  -(u.y*P(2,3)-P(1,3)),
		  -(u1.x*P1(2,3)-P1(0,3)),
		  -(u1.y*P1(2,3)-P1(1,3)));
    cv::Mat X;
    cv::solve(A,B,X,cv::DECOMP_SVD);
    return X;
}

/**
   @brief From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997

   @param u homogenious image point in 1st image plane
   @param P 1st camera matrix
   @param u1 homegenious image point in 2nd image plane
   @param P1 2nd camera matrix
*/
cv::Mat
IterativeLinearLSTriangulation(cv::Point3d u, cv::Matx34d P, cv::Point3d u1, cv::Matx34d P1)
{
    double wi = 1, wi1 = 1;
    cv::Matx41d X;
    for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most
	cv::Mat X_ = LinearLSTriangulation(u,P,u1,P1);
/*        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;
        
//recalculate weights
double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);
         
//breaking point
if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;
         
wi = p2x;
wi1 = p2x1;
         
//reweight equations and solve
Matx43d A((u.x*P(2,0)-P(0,0))/wi,       (u.x*P(2,1)-P(0,1))/wi,         (u.x*P(2,2)-P(0,2))/wi,    
(u.y*P(2,0)-P(1,0))/wi,       (u.y*P(2,1)-P(1,1))/wi,         (u.y*P(2,2)-P(1,2))/wi,    
(u1.x*P1(2,0)-P1(0,0))/wi1,   (u1.x*P1(2,1)-P1(0,1))/wi1,     (u1.x*P1(2,2)-P1(0,2))/wi1,
(u1.y*P1(2,0)-P1(1,0))/wi1,   (u1.y*P1(2,1)-P1(1,1))/wi1,     (u1.y*P1(2,2)-P1(1,2))/wi1
);
Mat_<double> B = (Mat_<double>(4,1) <<    -(u.x*P(2,3)    -P(0,3))/wi,
-(u.y*P(2,3)  -P(1,3))/wi,
-(u1.x*P1(2,3)    -P1(0,3))/wi1,
-(u1.y*P1(2,3)    -P1(1,3))/wi1
);
	
solve(A,B,X_,DECOMP_SVD);
X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;*/
    }
    return X;
}

//Triagulate points
void TriangulatePoints(const std::vector<Point2f>& pt_set1,
                       const std::vector<Point2f>& pt_set2,
                       const cv::Matx43d& Kinv,
                       const cv::Matx34d& P,
                       const cv::Matx34d& P1,
                       std::vector<Point3d>& pointcloud,
                       std::vector<Point2f>& correspImg1Pt)
{
#ifdef __SFM__DEBUG__
    vector depths;
#endif
 
    pointcloud.clear();
    correspImg1Pt.clear();
    
    cout << "Triangulating...";
    double t = getTickCount();
    unsigned int pts_size = pt_set1.size();
#pragma omp parallel for
    for (unsigned int i=0; i<pts_size; i++) {
        Point2f kp = pt_set1[i];
	Point3d u(kp.x,kp.y,1.0);
	Matx41 ray1 = Kinv*Matx31(u);
	Point2f kp1 = pt_set2[i];
	Point3d u1(kp1.x,kp1.y,1.0);
	Matx41 ray2 = Kinv * Matx31(u1);
	Mat X = IterativeLinearLSTriangulation(u,P,u1,P1);
 
//      if(X(2) > 6 || X(2) < 0) continue;
 
#pragma omp critical
        {
            pointcloud.push_back(Point3d(X(0),X(1),X(2)));
            correspImg1Pt.push_back(pt_set1[i]);
#ifdef __SFM__DEBUG__
            depths.push_back(X(2));
#endif
        }
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    std::cout << "Done." << std::endl;

#ifdef __SFM__DEBUG__
    {
        double minVal,maxVal;
        minMaxLoc(depths, &minVal, &maxVal);
        Mat tmp(240,320,CV_8UC3); //cvtColor(img_1_orig, tmp, CV_BGR2HSV);
        for (unsigned int i=0; i <pts_size; i++) {
	    double _d = MAX(MIN((pointcloud[i].z-minVal)/(maxVal-minVal),1.0),0.0);
	    circle(tmp, correspImg1Pt[i], 1, Scalar(255 * (1.0-(_d)),255,255), CV_FILLED);
	}
        cvtColor(tmp, tmp, CV_HSV2BGR);
        imshow("out", tmp);
        waitKey(0);
        destroyWindow("out");
    }
#endif
}

#endif
