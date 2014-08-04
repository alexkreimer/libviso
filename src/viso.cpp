#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/flann/flann.hpp>
#include <map>
#include <stdexcept>
#include <vector>
#include <ctime>

#include <ctype.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <time.h>

#include <boost/format.hpp>
#include <boost/thread/thread.hpp>
#include <boost/make_shared.hpp>
#include <boost/log/trivial.hpp>

#include "viso.h"
#include "estimation.h"

using namespace std;
using namespace boost;
using namespace cvflann;

using cv::Vec3i;
using cv::Vec4i;
using cv::DataType;
using cv::FM_RANSAC;

class Odometer
{
    const cv::FeatureDetector &detector;
    const cv::DescriptorExtractor &descriptor;
};

void
radiusSearch(cv::flann::Index& index, Mat& points, Mat& neighbors, float radius);

// each keypoint (x,y) is a raw in the resut matrix
Mat
kp2mat(const KeyPoints& kp)
{
    Mat mat(kp.size(), 2, DataType<float>::type);
    for(int i=0; i<kp.size(); ++i)
    {
        mat.at<float>(i,0) = kp.at(i).pt.x;
        mat.at<float>(i,1) = kp.at(i).pt.y;
    }
    return mat;
}

/* eucledian to homogenious */
MatrixXf e2h(const MatrixXf &xe)
{
    MatrixXf xh(xe.rows()+1, xe.cols());
    xh << xe, Eigen::MatrixXf::Constant(1, xe.cols(), 1.0);
    return xh;
}

MatrixXf h2e(const MatrixXf &xh)
{
    MatrixXf xe(xh.rows()-1, xh.cols());
    for(int i=0; i<xh.cols(); ++i)
    {
        if (abs(xh(xh.rows()-1,i)) < DBL_MIN)
            BOOST_LOG_TRIVIAL(error) << "h2e: divide by zero";
        xe.col(i) = xh.col(i).topRows(xh.rows()-1)/xh(xh.rows()-1,i);
    }
    return xe;
}

void
drawPoints(Mat& im, const MatrixXf& x, const Scalar& color, int thickness, int linetype)
{
    for(int i=0; i<x.cols(); ++i)
        circle(im, cv::Point(x(0,i), x(1,i)), thickness, color, linetype);
}

void
drawPoints(Mat& im, const KeyPoints& kp, int lim, const Scalar& color,
           int thickness, int linetype)
{
    for(int i=0; i<kp.size() && i<lim; ++i)
        circle(im, kp.at(i).pt, thickness, color, linetype);
}

void
save1(const Mat& im, const KeyPoints& kp, const string& file_name, int lim=INT_MAX,
      Scalar color=Scalar(255,0,0), int thickness=2, int linetype=-11)
{
    Mat im_rgb;
    cvtColor(im, im_rgb, CV_GRAY2RGB);
    drawPoints(im_rgb, kp, lim, color, thickness, linetype);
    cv::imwrite(file_name, im_rgb);
}

Eigen::MatrixXf
projectPoints(const MatrixXf& X, const MatrixXf P)
{
    return h2e(P*e2h(X));
}

void
showProjection(const Mat &im, const MatrixXf& x, const MatrixXf& X, const MatrixXf& P)
{
    Mat im_rgb;
    cvtColor(im, im_rgb, CV_GRAY2RGB);
    Scalar RED = Scalar(0,0,255), BLUE=Scalar(255,0,0);
    drawPoints(im_rgb, x, RED, 5, 1);
    drawPoints(im_rgb, projectPoints(X,P), BLUE, 3, -1);
    string title("Original points are red; reprojections are blue");
    cv::namedWindow(title);
    imshow(title, im_rgb);
    waitKey(0);
    cv::destroyWindow(title);
}

void
show2Projections(const Mat &im, const MatrixXf& X1, const MatrixXf& X2, const MatrixXf& P,
                 const string& file_name)
{
    Mat im_rgb;
    cvtColor(im, im_rgb, CV_GRAY2RGB);
    Scalar CYAN = Scalar(255, 255,0), GREEN = Scalar(0, 255, 0);
    drawPoints(im_rgb, projectPoints(X1,P), CYAN, 3, -1);
    drawPoints(im_rgb, projectPoints(X2,P), GREEN, 5, 1);
    string title("cyan are X1, green are X2");
    //cv::namedWindow(title);
    //imshow(title, im_rgb);
    //waitKey(0);
    //cv::destroyWindow(title);
    imwrite(file_name, im_rgb);
}

double
algebricDistance(const Mat& F, const Point2f& p1, const Point2f& p2)
{
    assert(F.type() == DataType<double>::type);
    float
        a0=p1.x, a1=p1.y, a2=1,
        b0=p2.x, b1=p2.y, b2=1;
    return
        b0*F.at<double>(0,0)*a0 +
        b0*F.at<double>(0,1)*a1 +
        b0*F.at<double>(0,2)*a2 +
        b1*F.at<double>(1,0)*a0 +
        b1*F.at<double>(1,1)*a1 +
        b1*F.at<double>(1,2)*a2 +
        b2*F.at<double>(2,0)*a0 +
        b2*F.at<double>(2,1)*a1 +
        b2*F.at<double>(2,2)*a2 ;
}

/*
  distance = sum (p2'Fp1)/n
 */
double
algebricDistance(const Mat& F, const Points2f& p1, const Points2f& p2)
{
    assert(p1.size()==p2.size());
    double err=0;
    for(Points2f::const_iterator i1=p1.begin(), i2=p2.begin();
        i1 != p1.end();
        ++i1, ++i2)
    {
        err += algebricDistance(F, *i1, *i2);
    }
    return err/p1.size();
}

set<int>
randomsample(int k, int n)
{
    BOOST_ASSERT_MSG(k<=n, (boost::format("k=%d,n=%d")%k%n).str().c_str());
    std::set<int> s;
    while(s.size()<k)
        s.insert(round(rand() % n));
    return s;
}

// T*X2 = X1
void
getRMS(const MatrixXf& X1, const MatrixXf& X2, const Affine3f& T,
       vector<int>& inliers, vector<float>& err, double& rms,
       double& inlier_rms, double thresh)
{
    assert(X1.rows()==3);
    assert(X2.rows()==3);
    assert(X1.cols()==X2.cols());
    MatrixXf X1h = e2h(X1), X2h = e2h(X2);
    MatrixXf X2h_rot = T.matrix()*X2h;
    MatrixXf delta = h2e(X1h)-h2e(X2h_rot);
    MatrixXf sq_err = delta.colwise().squaredNorm();
    inliers.clear();
    err.clear();
    for(int i=0; i<sq_err.cols(); ++i)
    {
        if (sq_err(i)<thresh*thresh)
        {
            inliers.push_back(i);
            err.push_back(sqrt(sq_err(i)));
        }
    }
    inlier_rms=0;
    for(auto &e: err)
    {
        inlier_rms += e*e;
    }
    inlier_rms = sqrt(inlier_rms/err.size());
    rms = sqrt((sq_err.array()/sq_err.cols()).sum());
}


/*
 * copy points from kp1, kp2 to p1,p2 according to match
 * \param kp1 source key points
 * \param kp2 source key points
 * \param match match info
 * \param p1 output
 * \param p2 output
 */
void
collect_matches(const KeyPoints& kp1, const KeyPoints &kp2,
                const Matches &match, Points2f &p1, Points2f &p2, int lim)
{
    p1.clear(); p2.clear();
    int i=0;
    for (auto &m : match)
    {
        if (i >=lim)
            break;
        p1.push_back(kp1.at(m[0]).pt);
        p2.push_back(kp2.at(m[1]).pt);
        ++i;
    }
}

/* concatenate a pair of matrices vertically
   m1, m2 must have the same number of columns
*/
void
save2(const cv::Mat& im1, const cv::Mat& im2, const KeyPoints& kp1,
      const KeyPoints& kp2, const Matches &match, const string& file_name,
      int lim)
{
    cv::Mat im_t = vcat<uchar>(im1, im2), im;
    cvtColor(im_t, im, CV_GRAY2RGB);
#if 0
    for (int i=0; i<kp1.size(); ++i)
    {
        circle(im, kp1.at(i).pt, 3, Scalar(255,255,0), -1);
    }
    for (int i=0; i<kp2.size(); ++i)
    {
        Point center(kp2.at(i).pt.x, kp2.at(i).pt.y+im1.rows);
        circle(im, center, 3, Scalar(255,250,0), -1);
    }
#endif
    for(int i=0; i<match.size() && i<lim; ++i)
    {
        Point 
            p1 = kp1[match.at(i)[0]].pt,
            p2 = kp2[match.at(i)[1]].pt;
        p2.y += im1.rows;
        line(im, p1, p2, Scalar(255));
    }
    imwrite(file_name, im);
}

void
save2epip(const cv::Mat& im1, const cv::Mat& im2, const Mat& F,
          Point2f pt, const KeyPoints& kp2, const string& file_name)
{
    cv::Mat im_t = vcat<uchar>(im2, im1), im;
    cvtColor(im_t, im, CV_GRAY2RGB);
    circle(im, Point(pt.x, pt.y+im2.rows), 3, Scalar(0,0,255), -1);
    for (int i=0; i<kp2.size(); ++i)
        circle(im, kp2.at(i).pt, 3, Scalar(255,0,0), -1);
    // draw the left points corresponding epipolar lines in right image 
    std::vector<cv::Vec3f> lines;
    vector<Point2f> pts;
    pts.push_back(pt);
    cv::computeCorrespondEpilines(pts, 1, F, lines);
    //for all epipolar lines
    for (vector<cv::Vec3f>::const_iterator it=lines.begin(); it!=lines.end(); ++it)
    {
        // draw the epipolar line between first and last column
        cv::line(im,cv::Point(0,-(*it)[2]/(*it)[1]),
                 cv::Point(im2.cols,-((*it)[2]+(*it)[0]*im2.cols)/(*it)[1]),
                 cv::Scalar(255,255,255));
    }
    imwrite(file_name, im);
}

void
save4(const Mat& im1, const Mat& im1_prev,
      const Mat& im2, const Mat& im2_prev,
      const KeyPoints& kp1, const KeyPoints& kp1_prev,
      const KeyPoints& kp2, const KeyPoints& kp2_prev,
      const vector<Vec4i>& ind, const string& file_name,
      int lim=INT_MAX)
{
    Mat im_t = vcat<uchar>(im1, im1_prev), im_t1 = vcat<uchar>(im2, im2_prev);
    Mat im_t2 = hcat<uchar>(im_t, im_t1);
    cv::Mat im;
    cvtColor(im_t2, im, CV_GRAY2RGB);
    Scalar magenta = Scalar(255, 255,0);
    Scalar green = Scalar(0, 255, 0);
    Scalar yellow = Scalar(0, 255, 255);
    Scalar red = Scalar(0, 0, 255);
    for(int i=0; i<ind.size() && i<lim; ++i)
    {
        Point 
            p1 = kp1[ind.at(i)[0]].pt,
            p2 = kp2[ind.at(i)[1]].pt,
            p3 = kp1_prev[ind.at(i)[2]].pt,
            p4 = kp2_prev[ind.at(i)[3]].pt;
        p2.x += im1.cols;
        p3.y += im1.rows;
        p4.y += im1.rows;
        p4.x += im1.cols;
        line(im, p1, p2, green);
        line(im, p2, p4, green);
        line(im, p4, p3, green);
        line(im, p3, p1, green);
    }
    imwrite(file_name, im);
}


/*
 * p2'Fp1 = 0
 */
double
sampsonDistance(const Mat& F,const Point2f& p1,const Point2f &p2)
{
    assert(F.type() == cv::DataType<double>::type);
    double 
        f0s = F.at<double>(0,0)*F.at<double>(0,0), 
        f1s = F.at<double>(1,0)*F.at<double>(1,0),
        f3s = F.at<double>(0,1)*F.at<double>(0,1);
    float x1s = p1.x*p1.x, y1s = p1.y*p1.y, x2s = p2.x*p2.x, y2s = p2.y*p2.y;
    return algebricDistance(F, p1, p2)/(f0s*x1s+f1s*y1s+f0s*x2s+f3s*y2s);
}
// match descriptors d1 vs. d2 using L2 distance
// and David Lowe 2nd best
void
match_l2_2nd_best_flann(const KeyPoints& kp1, const KeyPoints& kp2,
                        const cv::Mat& d1, const cv::Mat& d2,
                        Matches& match, float ratio=.65)
{
    // each row of d1 and d2 is a descriptor
    assert(d1.cols == d2.cols);
    match.clear();
    Mat mat_kp1 = kp2mat(kp1), mat_kp2 = kp2mat(kp2);
    // create kd-tree of kp2 locations
    cv::flann::Index tree(mat_kp2, cv::flann::KDTreeIndexParams(16));
    int MAX_NEIGHBORS=50;
    Mat neighbors(mat_kp1.rows, MAX_NEIGHBORS, DataType<int>::type, Scalar(-1));
    // search for neighbors. upon return neighbors.row(i) contains 
    // indices into kp2 of nearest neighbor points
    radiusSearch(tree, mat_kp1, neighbors, 30.0f);
    for(int i=0; i<kp1.size(); ++i)
    {
        float best_d1 = FLT_MAX, best_d2 = FLT_MAX;
        int best_idx = -1;
        for(int j=0; j<MAX_NEIGHBORS; ++j)
        {
            int cur_neighbor = neighbors.at<int>(i,j);
            if (cur_neighbor<0)
                break;
            double d = cv::norm(d2.row(cur_neighbor)-d1.row(i));
            if (d <= best_d1)
            {
                best_d2 = best_d1;
                best_d1 = d;
                best_idx = j;
            } else if (d <= best_d2) {
                best_d2 = d;
            }
        }
        // Lowe's 2nd best
        if (best_idx >= 0 && best_d1<best_d2*ratio)
        {
            match.push_back(Match(i, best_idx, best_d1));
        }
    }
    std::sort(match.begin(), match.end(), [](const Match &a, const Match &b) { return a[2]<b[2];});
}

// match descriptors d1 vs. d2 using L2 distance
// and David Lowe 2nd best
void
match_l2_2nd_best(const cv::Mat& d1,
                  const cv::Mat& d2,
                  Matches& match,
                  float ratio)
{
    // each row of d1 and d2 is a descriptor
    assert(d1.cols == d2.cols);
    match.clear();
    for(int i=0; i<d1.rows; ++i)
    {
        float best_d1 = FLT_MAX, best_d2 = FLT_MAX;
        int best_idx = -1;
        for(int j=0; j<d2.rows; ++j)
        {
            double d = cv::norm(d2.row(j)-d1.row(i));
            if (d <= best_d1)
            {
                best_d2 = best_d1;
                best_d1 = d;
                best_idx = j;
            } else if (d <= best_d2) {
                best_d2 = d;
            }
        }
        // Lowe's 2nd best
        if (best_idx >= 0 && best_d1<best_d2*ratio)
        {
            match.push_back(Match(i, best_idx, best_d1));
        }
    }
    std::sort(match.begin(), match.end(), [](const Match &a, const Match &b) { return a[2]<b[2];});
}

/* p2Fp1=0 */
void
match_epip_constraint(const cv::Mat& F, const KeyPoints& kp1, 
                      const KeyPoints& kp2, const Descriptors& d1,
                      const Descriptors& d2, Matches &match,
                      double ratio, double samp_thresh, double alg_thresh)
{
    match.clear();
    for(int i=0; i<kp1.size(); ++i)
    {
        Point2f p1 = kp1[i].pt;
        double best_d1 = DBL_MAX, best_d2 = DBL_MAX;
        pair<double,double> best_e;
        int best_idx = -1;
        for(int j=0; j<kp2.size(); ++j)
        {
            Point2f p2 = kp2[j].pt;
            pair<double,double> e = make_pair(sampsonDistance(F, p1, p2), samp_thresh);
            if (!std::isfinite(e.first))
                e = make_pair(algebricDistance(F, p1, p2), alg_thresh);
            double d = cv::norm(d2.row(j)-d1.row(i));
            if (d <= best_d1)
            {
                best_d2 = best_d1;
                best_d1 = d;
                best_idx = j;
                best_e = e;
            } else if (d <= best_d2)
                best_d2 = d;
        }
        // Lowe's 2nd best
        if (best_idx >= 0 && best_d1 < best_d2*ratio)
        {
            if (best_e.first < best_e.second)
                match.push_back(Match(i, best_idx, best_d1));
        }
    }
    std::sort(match.begin(), match.end(), [](const Match &a, const Match &b) { return a[2]<b[2];});
}

void
radiusSearch(cv::flann::Index& index, Mat& points, Mat& neighbors, float radius)
{
    assert(points.type()==DataType<float>::type);
    assert(neighbors.type()==DataType<int>::type);
    for(int i=0; i<points.rows; i++)
    {
        Mat p(1, points.cols, DataType<float>::type, points.ptr<float>(i)),
            n(1, neighbors.cols, CV_32SC1, neighbors.ptr<int>(i));
        Mat dist(1, neighbors.cols, DataType<float>::type);
        // neighbors is assumed to be inited to some invalid index value (e.g., -1)
        // so later on we can figure out how many neighbors were actually found
        int found = index.radiusSearch(p, n, dist, radius, neighbors.cols, cv::flann::SearchParams());
        cout << "found="<<found << ";" << "dist=" << _str<float>(dist) << endl;
        cout << "n=" << _str<int>(n) << endl;
        cout << "neighbors.ptr<int>(i)=" << _str<int>(neighbors.row(i)) << endl;
        cout << "p=" << _str<float>(p);
    }
}

void
radiusSearch2(cv::flann::Index& index, Mat& query, Mat& points2, Mat& neighbors, float radius)
{
    assert(query.type()==DataType<float>::type);
    assert(neighbors.type()==DataType<int>::type);
    int MAX_NEIGHBORS=neighbors.cols;
    for(int i=0; i<query.rows; i++)
    {
        Mat p(1, query.cols, DataType<float>::type, query.ptr<float>(i)),
            n(1, MAX_NEIGHBORS, DataType<float>::type, neighbors.ptr<int>(i)),
            dist(1, MAX_NEIGHBORS, DataType<double>::type);
        // neighbors is assumed to be inited to some invalid index value (e.g., -1)
        // so later on we can figure out how many neighbors were actually found
        int found = index.radiusSearch(p, n, dist, radius, MAX_NEIGHBORS, cv::flann::SearchParams());
        if (!found)
            continue;
        cout << "i=" << i << ";found="<<found << ";" << "dist=" << _str<float>(dist) << endl;
        cout << "n=" << _str<int>(n) << endl;
        cout << "neighbors.ptr<int>(i)=" << _str<int>(neighbors.row(i)) << endl;
        cout << "p=" << _str<float>(p);
        cout << "matches=";
        for(int j=0; j<found; ++j)
        {
            cout << "(ind=" << neighbors.at<int>(i,j) << ",";
            cout << _str<float>(points2.row(neighbors.at<int>(i,j))) << "), ";
        }
        cout << endl;
    }
}

void
radiusSearch3(Index<L2<float>>& index, Mat& query, Mat& points2, Mat& neighbors, float radius)
{
    assert(query.type()==DataType<float>::type);
    assert(neighbors.type()==DataType<int>::type);
    int MAX_NEIGHBORS=neighbors.cols;
    // construct an randomized kd-tree index using 4 kd-trees
    Matrix<float> dists(new float[neighbors.cols], 1, neighbors.cols); // dists
    for(int i=0; i<query.rows; i++)
    {
        Matrix<float> q(query.ptr<float>(i), 1, query.cols);  //query point
        Matrix<int> indices((int*)neighbors.ptr<int>(i), 1, neighbors.cols);  // neighbor indices
        int found = index.radiusSearch(q, indices, dists, radius, SearchParams(128));
        if (!found)
            continue;
        cout << "query point " << i << "; found=" << found << " neighbors;" << endl;
        cout << "neighbors.ptr<int>(i)=" << _str<int>(neighbors.row(i)) << endl;
        cout << "query point " << _str<float>(query.row(i));
        cout << "; matches=";
        for(int j=0; j<found && j<MAX_NEIGHBORS; ++j)
        {
            int ind = neighbors.at<int>(i,j);
            if (ind<0) break;
            cout << "(match index=" << ind << "," << "match distance=" << ((float *)dists.data)[j] << ",";
            cout << _str<float>(points2.row(ind)) << "), ";
        }
        cout << endl;
    }
    delete [] dists.data;
}

/* p2Fp1=0 */
void
match_epip_constraint_flann(const Mat& im1, const Mat& im2, const cv::Mat& F,
                            const KeyPoints& kp1, const KeyPoints& kp2,
                            const Descriptors& d1, const Descriptors& d2, 
                            Matches &match, double ratio, double samp_thresh,
                            double alg_thresh, int MAX_NEIGHBORS=100)
{
    // create kd-tree of kp2 locations
    //cv::flann::Index tree(kp2mat(kp2), cv::flann::KDTreeIndexParams(16));
    //cv::flann::Index linear(kp2mat(kp2), cv::flann::LinearIndexParams());
    Mat neighbors(kp1.size(), MAX_NEIGHBORS, DataType<int>::type, Scalar(-1));
    // search for neighbors. upon return neighbors.row(i) contains 
    // indices into kp2 of nearest neighbor points
    //radiusSearch(tree, mat_kp1, neighbors, 500.0f);
    Mat mat_kp1 = kp2mat(kp1), mat_kp2 = kp2mat(kp2);
    double radius = 50*50;
    //radiusSearch2(linear, mat_kp1, mat_kp2, neighbors, radius);
    match.clear();
    Matrix<float> dataset((float *)mat_kp2.data, (size_t)kp2.size(), (size_t)2), query((float*)mat_kp1.data, (size_t)kp1.size(), (size_t)2);
    // construct an randomized kd-tree index using 4 kd-trees
    //Index<L2<float> > index(dataset, KDTreeIndexParams(4));
    Index<L2<float> > index(dataset, LinearIndexParams());
    index.buildIndex();
    assert(index.size() == mat_kp2.rows);
    radiusSearch3(index, mat_kp1, mat_kp2, neighbors, radius);
    for(int i=0; i<kp1.size(); ++i)
    {
//        cout << "neighbors.row(i)=" << neighbors.row(i) << endl;
        Point2f p1 = kp1[i].pt;
        double best_d1 = DBL_MAX, best_d2 = DBL_MAX;
        pair<double,double> best_e;
        int best_idx = -1;
        KeyPoints tmp2;
        for(int j=0; j<MAX_NEIGHBORS; ++j)
        {
            int cur_neighbor_idx = neighbors.at<int>(i,j);
            if (cur_neighbor_idx<0) break; // no more neighbors
            Point2f p2 = kp2.at(cur_neighbor_idx).pt;
            tmp2.push_back(kp2.at(cur_neighbor_idx));
            pair<double,double> e = make_pair(sampsonDistance(F, p1, p2), samp_thresh);
            if (!std::isfinite(e.first))
                e = make_pair(algebricDistance(F, p1, p2), alg_thresh);
            if (best_e.first > best_e.second)
                continue;
            double d = cv::norm(d2.row(cur_neighbor_idx)-d1.row(i), cv::NORM_L1);
            if (d <= best_d1)
            {
                best_d2 = best_d1;
                best_d1 = d;
                best_idx = cur_neighbor_idx;
                best_e = e;
            } else if (d <= best_d2)
                best_d2 = d;
        }
        if (tmp2.size()>0)
            save2epip(im1, im2, F, p1, tmp2, (boost::format("epip_match_%03d.jpg")%i).str().c_str());
        // Lowe's 2nd best
        if (best_idx >= 0 /*&& best_d1 < best_d2*ratio*/)
        {
            match.push_back(Match(i, best_idx, best_d1));
        }
    }
    std::sort(match.begin(), match.end(), [](const Match &a, const Match &b) { return a[2]<b[2];});
}

/*
 * match is a vector of pairs that store keypoint correspondences
 * Let m = match[k], then m[0] is the index of keypoint in the first view,
 * m[1] is the index of a matching kp in the 2nd view.
 *
 * has_match checks wheter keypoint with index=kpi in the view=view has a
 * match in the 2nd view
 */
int
kp_has_match(const Matches& match, int view, int kpi)
{
    int val = -1;
    for(auto &m: match)
    {
        if (m[view] == kpi)
        {
            val = m[(view+1)%2];
        }
    }
    return val;
}

/* kp_match checks if kpi1 and kpi2 match each other
 */
int
kp_match(const Matches& match, int kpi1, int kpi2)
{
    for(int i=0; i<match.size(); ++i)
    {
        if (match[i][0] == kpi1 && match[i][1] == kpi2)
        {
            return i;
        }
    }
    return -1;
}

class CustomFeatureDetector : public cv::FeatureDetector
{
protected:
    void detectImpl(const Mat& image, std::vector<cv::KeyPoint>& kp, const Mat& mask=Mat()) const {}
};

void
myhist(const Mat& image)
{
    double min_val, max_val;
    minMaxLoc(image, &min_val, &max_val, 0, 0);

    Mat hist;
    int hist_size=300;
    float range[] = { (float)min_val, (float)max_val} ;
    const float* hist_range = {range};
    bool uniform = true; bool accumulate = false;
    cv::calcHist(&image, 1 /*images num in prev arg*/, 0 /*channels*/, Mat() /*mask*/,
                 hist, 1, &hist_size, &hist_range, uniform, accumulate);
    int hist_w = 1024, hist_h = 800, bin_w = cvRound((double) hist_w/hist_size);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar( 0,0,0));
    cout << "histogram values: ";
    for(int i=1; i<hist_size; i++)
    {
        cv::line(histImage, cv::Point(bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1))),
                 cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                 cv::Scalar( 255, 0, 0), 2, 8, 0  );
        cout << cvRound(hist.at<float>(i)) << ",";
    }
    cout << endl;
    /// Display
    cv::namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    cv::imshow("calcHist Demo", histImage );
    cv::waitKey(0);
}

void
saveHarrisCorners(const Mat& harris_response, int thresh, const string& file_name)
{
    Mat dst, dst_norm, dst_norm_scaled;
    Scalar RED = Scalar(0,0,255), color=RED;
    int thickness=5, linetype=1;
    cv::normalize(harris_response, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    Mat im_rgb;
    cvtColor(dst_norm_scaled, im_rgb, CV_GRAY2RGB);
    for( int j = 0; j < dst_norm_scaled.rows ; j++ )
    {
        for( int i = 0; i < dst_norm_scaled.cols; i++ )
        {
            if ((int)dst_norm.at<float>(j,i)>thresh)
            {
                circle(im_rgb, Point(i,j), thickness, color, linetype);
            }
        }
    }
    cv::imwrite(file_name, dst_norm_scaled);
}

void
showHarris(const Mat& harris_response, int thresh)
{
    Mat dst, dst_norm, dst_norm_scaled;
    cv::normalize(harris_response, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    /// Drawing a circle around corners
    for( int j = 0; j < dst_norm_scaled.rows ; j++ )
    {
        for( int i = 0; i < dst_norm_scaled.cols; i++ )
        {
            if ((int)dst_norm.at<float>(j,i)>thresh)
            {
                circle(dst_norm_scaled, Point(i, j), 3,  Scalar(0), 2, 8, 0);
            }
        }
    }
    /// Showing the result
    cv::namedWindow("win", CV_WINDOW_AUTOSIZE );
    cv::imshow("win", dst_norm_scaled);
    cv::waitKey(0);
}

class HarrisFeatureDetector : public cv::FeatureDetector
{
public:
    // descriptor radius is used only to init KeyPoints
    HarrisFeatureDetector(int descriptor_radius) 
        : m_descriptor_radius(descriptor_radius) {}

protected:
    void
    detectImpl(const Mat& image, vector<KeyPoint>& kp, const Mat& mask=Mat()) const
    {
        Mat dst, dst_norm;
        dst = Mat::zeros(image.size(), DataType<float>::type);
        int block_size=3, aperture_size=5, thresh=80;
        double k=.04; // M_c = det(A) - k*trace^2(A), the range for k \in [0.04, 0.15]
        cv::cornerHarris(image, dst, block_size, aperture_size, k, cv::BORDER_DEFAULT);
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, Mat());
        for(int j=0; j<dst_norm.rows ; j++ )
        {
            for( int i = 0; i<dst_norm.cols; i++ )
            {
                if ((int)dst_norm.at<float>(j,i)>thresh)
                {
                    kp.push_back(KeyPoint(Point2f(i,j), 2*m_descriptor_radius+1));
                }
            }
        }

    }
private:
    int m_descriptor_radius;
};

class MyFeatureExtractor : public cv::DescriptorExtractor
{
public:
    MyFeatureExtractor(int descriptor_radius) 
        : m_descriptor_radius(descriptor_radius) {}
protected:
    int m_descriptor_radius;

    int descriptorType() const
    {
        return DataType<float>::type;
    }

    int 
    descriptorSize() const
    {
        return (2*m_descriptor_radius+1)*(2*m_descriptor_radius+1);
    }

    /* TODO: check border */
    void
    computeImpl(const Mat& image, vector<KeyPoint>& kp, Mat& d) const
    {
        Mat sob(image.rows, image.cols, CV_32F, Scalar(0));
        d = Mat(kp.size(), descriptorSize(), DataType<float>::type, Scalar(0));
        Sobel(image, sob, sob.type(), 1, 0, 3, 1, 0, cv::BORDER_REFLECT_101);
        for(int k=0; k<kp.size(); ++k)
        {
            Point2i p = kp.at(k).pt;
            for(int i=-m_descriptor_radius,col=0; i<m_descriptor_radius; ++i)
            {
                for(int j=-m_descriptor_radius; j<m_descriptor_radius; ++j,++col)
                {
                    d.at<float>(k,col) = 
                        (p.y+i>0 && p.y+i<image.rows && p.x+j>0 && p.x+j<image.cols) ?
                        sob.at<float>(p.y+i, p.x+j) : 0;
                }
            }
        }
    }
};

typedef Mat Descriptor;

void
localMatch(const KeyPoints& kp1, const KeyPoints& kp2,
           const Descriptor& d1, const Descriptor& d2,
           Matches& match, double thresh)
{
    double min_d = DBL_MAX;
    Match best_match;
    for(int i=0; i<kp1.size(); ++i)
    {
        for(int j=0; j<kp2.size(); ++j)
        {
            double d = norm(kp1.at(i).pt-kp2.at(j).pt);
            if (d>thresh)
                continue;
            d = cv::norm(d1.col(i)-d2.col(j));
            if (d<min_d)
            {
                best_match = Match(i,j);
            }
        }
        match.push_back(best_match);
    }
}

// stereo odometry

vector<Affine3f>
sequenceOdometry(const Mat& P1, const Mat& P2, StereoImageGenerator& images)
{
    MatrixXf eP1; cv2eigen(P1, eP1);
//    cv::SiftFeatureDetector detector;
    HarrisFeatureDetector detector(5);
    MyFeatureExtractor extractor(5);
    StereoImageGenerator::result_type stereo_pair;
    Mat F = F_from_P<double>(P1,P2);
    if (F.at<double>(2,2) > DBL_MIN)
    {
        F /= F.at<double>(2,2);
    }
    BOOST_LOG_TRIVIAL(info) << (boost::format("P1=%s, P2=%s, F=%s") % _str<double>(P1) % _str<double>(P2) % _str<double>(F)).str();
    // reconstructed 3d clouds
    Mat X, X_prev;
    // current and previous pair of images
    Mat im1, im1_prev, im2, im2_prev;
    // current and previous set of descriptors
    Mat d1, d2, d1_prev, d2_prev;
    // current and previous set of keypoints
    KeyPoints kp1, kp2, kp1_prev, kp2_prev;
    // current and previous matches (for the stereo pair images)
    Matches match, match_prev;
    // result
    vector<Affine3f> poses;
    bool first = true;
    const clock_t begin_time = clock();
    int iter_num;
    for(iter_num=0; (stereo_pair=images()); ++iter_num)
    {
        BOOST_LOG_TRIVIAL(info) << "iter: " << iter_num;
        if (!first) 
        {
            im1.copyTo(im1_prev);
            im2.copyTo(im2_prev);
            d1.copyTo(d1_prev);
            d2.copyTo(d2_prev);
            kp1_prev = kp1;
            kp2_prev = kp2;
            match_prev = match;
            X.copyTo(X_prev);
            kp1.clear();
            kp2.clear();
            match.clear();
        }
        im1 = (*stereo_pair).first;
        im2 = (*stereo_pair).second;
        assert(im1.data && im2.data);
        detector.detect(im1,kp1);
	detector.detect(im2,kp2);
	BOOST_LOG_TRIVIAL(info) << kp1.size() << " keypoints found in the 1st image";
	BOOST_LOG_TRIVIAL(info) << kp2.size() << " keypoints found in the 2nd image";
        extractor.compute(im1, kp1, d1);
	extractor.compute(im2, kp2, d2);
        save1(im1, kp1, (boost::format("left_%03d.jpg") % iter_num).str().c_str(), INT_MAX);
        save1(im2, kp2, (boost::format("right_%03d.jpg") % iter_num).str().c_str(), INT_MAX);
        match_epip_constraint_flann(im1, im2, F, kp1, kp2, d1, d2, match, 0.8 /*2nd best ratio */, 1 /*sampson distance*/, .5 /*algebric error*/);
        BOOST_LOG_TRIVIAL(debug) << cv::format("Done matching left vs right: %d matches", match.size());
        save2(im1, im2, kp1, kp2, match, (boost::format("lr_%03d.jpg") % iter_num).str().c_str());
        exit(1);
	cv::Mat x1(2, match.size(), CV_32FC1), x2(2, match.size(), CV_32FC1);
	for(int i=0; i<match.size(); i++)
	{
	    int i1 = match.at(i)[0], i2 = match.at(i)[1];
            x1.at<float>(0, i) = kp1.at(i1).pt.x;
            x1.at<float>(1, i) = kp1.at(i1).pt.y;
            x2.at<float>(0, i) = kp2.at(i2).pt.x;
            x2.at<float>(1, i) = kp2.at(i2).pt.y;
	}
        X = triangulate_dlt(x1, x2, P1, P2);
        //Eigen::MatrixXf eX, ex1; cv2eigen(x1, ex1); cv2eigen(X,eX);
        //showProjection(im1, ex1, eX, eP1);
        if (first)
        {
            first = false;
            continue;
        }
        Matches match11, match12, match22, match21;
        Mat F11, F22, F12, F21;
        Points2f p1, p1_prev, p2, p2_prev;
        match_l2_2nd_best_flann(kp1, kp1_prev, d1, d1_prev, match11);
        //show2(im1, im1_prev, kp1, kp1_prev, match11, "Left view match: current vs prev (2nd best)");
        collect_matches(kp1, kp1_prev, match11, p1, p1_prev);
        F11 = findFundamentalMat(p1, p1_prev, cv::FM_RANSAC, 3, .99999);
        match_epip_constraint(F11, kp1, kp1_prev, d1, d1_prev, match11, .8, 3, .1);
        save2(im1, im1_prev, kp1, kp1_prev, match11,
              (boost::format("ll_%s.jpg")%iter_num).str().c_str(), 15);
        BOOST_LOG_TRIVIAL(debug) << cv::format("Done matching left vs left_prev: %d matches", match11.size());
        assert(F11.type()==6);
        match_l2_2nd_best_flann(kp2, kp2_prev, d2, d2_prev, match22);
        collect_matches(kp2, kp2_prev, match22, p2, p2_prev);
        F22 = findFundamentalMat(p2, p2_prev, cv::FM_RANSAC);
        match_epip_constraint(F22, kp2, kp2_prev, d2, d2_prev, match22, .8, 2, .5);
        save2(im2, im2_prev, kp2, kp2_prev, match22,
              (boost::format("rr_%s.jpg")%iter_num).str().c_str(), 15);
        BOOST_LOG_TRIVIAL(debug) << cv::format("Done matching right vs right_prev: %d matches", match22.size());
        assert(F22.type()==6);
        //cout << "F22:" << _str<double>(F22) << "; mean algebric error=" << algebricDistance(F22, p2, p2_prev) << endl;
        // matches of 3d points
        Matches match_pcl; 
        // circular match
        vector<Vec4i> circ_match;
        for(int i=0; i<match.size(); ++i)
        {
            int kpi1 = match[i][0],
                kpi2 = match[i][1], 
                kpi1_prev = kp_has_match(match11, 0, kpi1),
                kpi2_prev = kp_has_match(match22, 0, kpi2);
            if (kpi1_prev>0 && kpi2_prev>0)
            {
                int i_prev = kp_match(match_prev, kpi1_prev, kpi2_prev);
                if (i_prev>0)
                {
                    circ_match.push_back(Vec4i(kpi1, kpi2, kpi1_prev, kpi2_prev));
                    match_pcl.push_back(Match(i, i_prev));
                }
            }
        }
        MatrixXf eigX(3, match_pcl.size()), eigX_p(3, match_pcl.size());
        assert(X.type() == DataType<float>::type);
        assert(X_prev.type() == DataType<float>::type);
        for(int i=0; i<match_pcl.size(); ++i)
        {
            int ind1 = match_pcl.at(i)[0], ind2 = match_pcl.at(i)[1];
            eigX(0,i) = X.at<float>(0,ind1);
            eigX(1,i) = X.at<float>(1,ind1);
            eigX(2,i) = X.at<float>(2,ind1);
            eigX_p(0,i) = X_prev.at<float>(0,ind2);
            eigX_p(1,i) = X_prev.at<float>(1,ind2);
            eigX_p(2,i) = X_prev.at<float>(2,ind2);
        }
        save4(im1, im1_prev, im2, im2_prev, kp1, kp1_prev, kp2, kp2_prev, circ_match,
              (boost::format("circ_match_%03d.jpg") % iter_num).str().c_str());
        BOOST_LOG_TRIVIAL(info) << match_pcl.size() << " points in circular match";
        BOOST_LOG_TRIVIAL(info) << "solving rigid motion" << endl;
        if (match_pcl.size()>=3) 
        {
            //number of tries; number of points needed to estimate a model (rotation,translation matrix)
            int N=10, model_pts_num = 3, ninliers=0;
            double min_rms = DBL_MAX, rms=DBL_MAX;
            vector<int> inliers;
            vector<float> err;
            Affine3f T_best;
            vector<Vec4i> circ_match_best;

            for(int k=0; k<N; ++k)
            {
                set<int> s = randomsample(model_pts_num, match_pcl.size());
                Eigen::MatrixXf X1(3, model_pts_num), X2(3, model_pts_num);
                vector<Vec4i> circ_match_sample;
                int i=0;
                for(auto j: s)
                {
                    circ_match_sample.push_back(circ_match.at(j));
                    X1.col(i) = eigX.col(j);
                    X2.col(i) = eigX_p.col(j);
                    ++i;
                }
                Eigen::Affine3f T;
                //T*X2 = X1
                solveRigidMotion(X1, X2, T);
                double sample_rms, inlier_rms, t=.2;
                inliers.clear();
                for(int i=0; i<3 && inliers.size()<3; ++i, t*=2)
                {
                    getRMS(eigX, eigX_p, T, inliers, err, rms, inlier_rms, t /*thresh*/);
                }
                if (ninliers<inliers.size())
                {
                    ninliers = inliers.size();
                    T_best = T;
                    circ_match_best = circ_match_sample;
                }
            }
            poses.push_back(T_best);
            show2Projections(im1, eigX, h2e(T_best.matrix()*e2h(eigX_p)), eP1,
                             (boost::format("reproj_%03d.jpg") % iter_num).str().c_str());
        } else {
            // need something else
            cout << "ERROR: not enough matches in circular match" << endl;
            break;
        }
    }
    BOOST_LOG_TRIVIAL(debug) << "avg time per iteration [s]:" << float(clock()-begin_time)/CLOCKS_PER_SEC/iter_num << endl;
    return poses;
}
