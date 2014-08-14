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

using cv::Vec3i;
using cv::Vec4i;
using cv::Vec2f;
using cv::DataType;
using cv::FM_RANSAC;
using cv::InputArray;
using cv::OutputArray;

class Odometer
{
    const cv::FeatureDetector &detector;
    const cv::DescriptorExtractor &descriptor;
};

struct MatchParams
{
    bool enforce_epipolar;
    Mat F;
    double alg_thresh;
    double sampson_thresh;

    bool enforce_2nd_best;
    double ratio_2nd_best;

    bool allow_ann;
    int max_neighbors;
    double radius;

    MatchParams(Mat F) : enforce_epipolar(true),
                         sampson_thresh(1),
                         enforce_2nd_best(false),
                         ratio_2nd_best(.8),
                         allow_ann(true),
                         max_neighbors(200),
                         radius(80)
        {
            F.copyTo(this->F);
        };
    MatchParams() : enforce_epipolar(false), enforce_2nd_best(true),
                    ratio_2nd_best(.9), allow_ann(true), max_neighbors(250),
                    radius(80) {}
};

MatrixXf
get_inl(const MatrixXf& X, const vector<int>& inl)
{
    assert(inl.size()>0 && inl.size()<X.cols());
    MatrixXf Xinl(X.rows(), inl.size());
    int j=0;
    for(auto i: inl)
    {
        Xinl.col(j++) = X.col(i);
    }
    return Xinl;
}
void
knnSearch(cv::flann::Index& index, Mat& query_points, Mat& neighbors, Mat& p2)
{
    assert(query_points.type()==DataType<float>::type);
    assert(neighbors.type()==DataType<int>::type);
    cout << "doing knnSearch, nn=" << neighbors.cols << endl;
    for(int i=0; i<query_points.rows; i++)
    {
        Mat query(1, query_points.cols, DataType<float>::type, query_points.ptr<float>(i)),
            neigh(1, neighbors.cols, CV_32SC1, neighbors.ptr<int>(i)),
            dists(1, neighbors.cols, DataType<float>::type);
        /*int found = */index.knnSearch(query, neigh, dists, neighbors.cols, cv::flann::SearchParams(128));
        for(int j=0; j<neighbors.cols; ++j)
        {
            int ind = neighbors.at<float>(i,j);
            if (ind<0) break;
            Mat pt2 = p2.row(ind);
            double dist = cv::norm(query-pt2);
            cout << "radius violation: p1=" << _str<float>(query) << "; p2=" << _str<float>(pt2) << "; dist=" << dist << "dist1=" << dists.at<float>(j) << endl;
        }
        cout << endl;
    }
}
void
radiusSearch(cvflann::Index<cvflann::L1<float>>& index, Mat& query_points, 
             Mat& neighbors, float radius, Mat& p2, bool dbg=1)
{
    assert(query_points.type()==DataType<float>::type);
    assert(neighbors.type()==DataType<int>::type);
    cvflann::Matrix<float> dists(new float[neighbors.cols], 1, neighbors.cols);
    for(int i=0; i<query_points.rows; i++)
    {
        cvflann::Matrix<float> query(query_points.ptr<float>(i), 1, query_points.cols);
        cvflann::Matrix<int> nei(neighbors.ptr<int>(i), 1, neighbors.cols);
        int found = index.radiusSearch(query, nei, dists, radius, nei.cols);
        for(int j=found; j<nei.cols; ++j)
        {
            nei.data[j] = -1;
            dists.data[j] = -1;
        }
#if 0
        Mat q(1, query_points.cols, DataType<float>::type, query.data),
            d(1, neighbors.cols, DataType<float>::type, dists.data);
        if (dbg)
            cout << "found=" << found << "; q=" << q << "; d=" << d << endl;
        for (int j=0; dbg && j<found && j<neighbors.cols; ++j)
        {
            cout << "nei=" << nei.data[j] << "; pt=[" << p2.at<float>(nei.data[j],0) <<"," << p2.at<float>(nei.data[j],1) << "]" 
                 << abs(query.data[0]-p2.at<float>(nei.data[j],0))+ abs(query.data[1]-p2.at<float>(nei.data[j],1)) << endl;
            if (abs(query.data[0]-22) < 5 && abs(query.data[1]-22)<5)
            {
                cout << "kuku";
            }
        }
#endif        
    }
}


void
match_circle(const Matches& match_lr, const Matches& match_lr_prev,
             const Matches& match11, const Matches& match22,
             vector<Vec4i>& circ_match, Matches& match_pcl)
{
    // iterate over left features that have a match
    for(int i=0; i<match_lr.size(); ++i)
    {
        int ileft = match_lr.at(i)[0], iright = match_lr.at(i)[1];
        // go over all matches to left prev
        for(int j=0; j<match11.size(); ++j)
        {
            if (match11.at(j)[0] == ileft)
            {
                int ileft_prev = match11.at(j)[1];
                for(int k=0; k<match_lr_prev.size(); ++k)
                {
                    if (match_lr_prev.at(k)[0] == ileft_prev)
                    {
                        int iright_prev = match_lr_prev.at(k)[1];
                        for(int l=0; l<match22.size(); ++l)
                        {
                            if (match22.at(l)[1] == iright_prev)
                            {
                                if (match22.at(l)[0] == iright)
                                {
                                    circ_match.push_back(Vec4i(ileft, iright, ileft_prev, iright_prev));
                                    match_pcl.push_back(Match(i,k));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

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
MatrixXf
e2h(const MatrixXf &xe)
{
    MatrixXf xh(xe.rows()+1, xe.cols());
    //xh << xe, Eigen::MatrixXf::Constant(1, xe.cols(), 1.0);
    for(int i=0; i<xe.rows(); ++i)
        for(int j=0; j<xe.cols(); ++j)
            xh(i,j) = xe(i,j);

    for(int j=0; j<xe.cols(); ++j)
        xh(xe.rows(),j) = 1.0;

    return xh;
}

MatrixXf
h2e(const MatrixXf &xh)
{
    MatrixXf xe(xh.rows()-1, xh.cols());
    for(int i=0; i<xh.cols(); ++i)
    {
        if (abs(xh(xh.rows()-1,i)) < DBL_MIN)
            throw std::overflow_error("divide by zero in h2e");
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
drawPoints(Mat& im, const Mat& x, const Scalar& color, int thickness, int linetype)
{
    assert(x.type() == DataType<float>::type);
    for(int i=0; i<x.cols; ++i)
        circle(im, cv::Point(x.at<float>(0,i), x.at<float>(1,i)), thickness, color, linetype);
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
      Scalar color=Scalar(255,0,0), int thickness=1, int linetype=-1)
{
    Mat im_rgb;
    cvtColor(im, im_rgb, CV_GRAY2RGB);
    drawPoints(im_rgb, kp, lim, color, thickness, linetype);
    cv::imwrite(file_name, im_rgb);
}

void
projectPoints(const MatrixXf& X, const MatrixXf P, MatrixXf& x)
{
    x = h2e(P*e2h(X));
}

Mat
projectPoints(const Mat& X, const Mat& P)
{
    assert(X.type()==DataType<float>::type);
    Mat Xh = e2h<float>(X), Pf;
    P.convertTo(Pf, DataType<float>::type);
    return h2e<float>(Pf*Xh);
}

void
showProjection(const Mat &im, const MatrixXf& x, const MatrixXf& X, const MatrixXf& P)
{
    Mat im_rgb;
    cvtColor(im, im_rgb, CV_GRAY2RGB);
    Scalar RED = Scalar(0,0,255), BLUE=Scalar(255,0,0);
    drawPoints(im_rgb, x, RED, 5, 1);                          
    MatrixXf x1;
    projectPoints(X,P,x1);
    drawPoints(im_rgb, x1, BLUE, 3, -1);
    string title("Original points are red; reprojections are blue");
    cv::namedWindow(title);
    imshow(title, im_rgb);
    waitKey(0);
    cv::destroyWindow(title);
}

void
save2reproj(const Mat &im, const MatrixXf& X1, const MatrixXf& X2,
            const MatrixXf& P, const string& file_name)
{
    Mat im_rgb;
    cvtColor(im, im_rgb, CV_GRAY2RGB);
    MatrixXf x1, x2;
    projectPoints(X1,P,x1);
    projectPoints(X2,P,x2);
    drawPoints(im_rgb, x1, Scalar(0,0,255), 2, -1);
    drawPoints(im_rgb, x2, Scalar(0,255,0), 3, 1);
    imwrite(file_name, im_rgb);
}

void
save2reproj(const Mat &im, const MatrixXf& X1, const MatrixXf& X2,
            const MatrixXf& P1, const MatrixXf& P2, const string& file_name)
{
    Mat im_rgb;
    cvtColor(im, im_rgb, CV_GRAY2RGB);
    MatrixXf x1,x2;
    projectPoints(X1,P1,x1);
    projectPoints(X2,P2,x2);
    drawPoints(im_rgb, x1, Scalar(0,0,255), 3, 1);
    drawPoints(im_rgb, x2, Scalar(0,255,0), 3, 1);
    imwrite(file_name, im_rgb);
}

bool
save1reproj(const Mat& im, const Mat& X,const Mat& x, const Mat& P, const string& file_name)
{
    cv::Mat im_rgb;
    cvtColor(im, im_rgb, CV_GRAY2RGB);
    drawPoints(im_rgb, x, Scalar(255,0,0), 1, -1);
    drawPoints(im_rgb, projectPoints(X, P), Scalar(0,255,0), 3, 1);
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
double
getRMS(const MatrixXf& X1, const MatrixXf& X2, const Affine3f& T,
       vector<int>& inliers, double thresh)
{
    assert(X1.rows()==3 && X2.rows()==3);
    assert(X1.cols()==X2.cols());

    MatrixXf er = X1-h2e(T.matrix()*e2h(X2));
    MatrixXf er_sq = er.colwise().squaredNorm();
    inliers.clear();
    double rms=0;
    for(int i=0; i<er_sq.cols(); ++i)
    {
        if (er_sq(i)<thresh*thresh)
        {
            inliers.push_back(i);
            rms += er_sq(i);
        }
    }
    rms /= inliers.size();
    return sqrt(rms);
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

void
collect_matches(const KeyPoints& kp1, const KeyPoints &kp2,
                const Matches &match, Mat &p1, Mat &p2)
{
    for(int i=0; i<match.size(); ++i)
    {
        int i1 = match.at(i)[0], i2 = match.at(i)[1];
        p1.at<float>(0,i) = kp1.at(i1).pt.x;
        p1.at<float>(1,i) = kp1.at(i1).pt.y;
        p2.at<float>(0,i) = kp2.at(i2).pt.x;
        p2.at<float>(1,i) = kp2.at(i2).pt.y;
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
    for (int i=0; i<kp1.size(); ++i)
    {
        circle(im, kp1.at(i).pt, 2, Scalar(255,0,0), -1);
    }
    for (int i=0; i<kp2.size(); ++i)
    {
        circle(im, Point(kp2.at(i).pt.x, kp2.at(i).pt.y+im1.rows), 2, Scalar(255,0,0), -1);
    }
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
save2blend(const cv::Mat& im1, const cv::Mat& im2, const KeyPoints& kp1,
           const KeyPoints& kp2, const Matches &match, const string& file_name,
           int lim=INT_MAX)
{
    cv::Mat blend, im;
    addWeighted(im1, .5, im2, .5, 0.0, blend);
    cvtColor(blend, im, CV_GRAY2RGB);
    for (int i=0; i<kp1.size(); ++i)
    {
//        circle(im, kp1.at(i).pt, 2, Scalar(255,0,0), -1);
    }
    for (int i=0; i<kp2.size(); ++i)
    {
//        circle(im, kp2.at(i).pt, 2, Scalar(0,255,0), -1);
    }
    for(int i=0;i<match.size() && i<lim; ++i)
    {
        Point 
            p1 = kp1[match.at(i)[0]].pt,
            p2 = kp2[match.at(i)[1]].pt;
        circle(im, p1, 1, Scalar(0,255,0), -1);
        line(im, p1, p2, Scalar(255,0,0));
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
        Fx0 = F.at<double>(0,0)*p1.x + F.at<double>(0,1)*p1.y + F.at<double>(0,2),
        Fx1 = F.at<double>(1,0)*p1.x + F.at<double>(1,1)*p1.y + F.at<double>(1,2),
        Ftx0= F.at<double>(0,0)*p2.x + F.at<double>(1,0)*p2.y + F.at<double>(2,0),
        Ftx1= F.at<double>(0,1)*p2.x + F.at<double>(1,1)*p2.y + F.at<double>(2,1);
    float ad = algebricDistance(F, p1, p2);
    return ad*ad/(Fx0*Fx0+Fx1*Fx1+Ftx0*Ftx0+Ftx1*Ftx1);
}

/* p2Fp1=0 */
void
match_desc(const KeyPoints& kp1, const KeyPoints& kp2,
           const Descriptors& d1, const Descriptors& d2,
           Matches &match, const MatchParams& sp = MatchParams())
{
    const clock_t begin_time = clock();
    match.clear();
    BOOST_ASSERT_MSG(d1.cols==d2.cols, (boost::format("d1.cols=%d,d2.cols=%d") % d1.cols % d2.cols).str().c_str());
    // cv::flann::Index index = (sp.allow_ann) ? 
    //     cv::flann::Index(kp2mat(kp2), cv::flann::KDTreeIndexParams(16), ::cvflann::FLANN_DIST_L1) :
    //     cv::flann::Index(kp2mat(kp2), cv::flann::LinearIndexParams(), ::cvflann::FLANN_DIST_L1);
    Mat kp1m = kp2mat(kp1), kp2m = kp2mat(kp2), neighbors(kp1m.rows, sp.max_neighbors,
                                                          DataType<int>::type, Scalar(-1));
    cvflann::Matrix<float> dataset((float*)kp2m.data, (size_t)kp2m.rows, (size_t)kp2m.cols);
    //cvflann::Index<cvflann::L1<float>> index(dataset, cvflann::KDTreeIndexParams(16));
    cvflann::Index<cvflann::L1<float>> index(dataset, cvflann::LinearIndexParams());
    radiusSearch(index, kp1m, neighbors, sp.radius, kp2m, !sp.enforce_epipolar);
    for(int i=0; i<kp1.size(); ++i)
    {
        Point2f p1 = kp1.at(i).pt;
        double best_d1 = DBL_MAX, best_d2 = DBL_MAX;
        pair<double,double> best_e;
        int best_idx = -1;
        for(int j=0, nind=neighbors.at<int>(i,j);
            j<neighbors.cols && nind>0; ++j, nind=neighbors.at<int>(i,j))
        {
            if (sp.enforce_epipolar)
            {
                Point2f p2 = kp2.at(nind).pt;
                double sampson_dist = sampsonDistance(sp.F, p1, p2);
                if (!std::isfinite(sampson_dist) || sampson_dist > sp.sampson_thresh)
                    continue;
            }
            double d = cv::norm(d2.row(nind)-d1.row(i), cv::NORM_L1);
            if (d <= best_d1)
            {
                best_d2 = best_d1;
                best_d1 = d;
                best_idx = nind;
            } else if (d <= best_d2)
                best_d2 = d;
        }
        if (best_idx >= 0)
        {
            if (sp.enforce_2nd_best)
            {
                if (best_d1 < best_d2*sp.ratio_2nd_best)
                {
                    match.push_back(Match(i, best_idx, best_d1));
                }
            } else {
                match.push_back(Match(i, best_idx, best_d1));
            }
        }
    }
    std::sort(match.begin(), match.end(), [](const Match &a, const Match &b) { return a[2]<b[2];});
    BOOST_LOG_TRIVIAL(info) << "match time [s]:" << float(clock()-begin_time)/CLOCKS_PER_SEC;
}

void
radiusSearch(cv::flann::Index& index, Mat& points, Mat& neighbors, float radius, Mat& p2)
{
    assert(points.type()==DataType<float>::type);
    assert(neighbors.type()==DataType<int>::type);
    for(int i=0; i<points.rows; i++)
    {
        Mat p(1, points.cols, DataType<float>::type, points.ptr<float>(i)),
            n(1, neighbors.cols, CV_32SC1, neighbors.ptr<int>(i)),
            dist(1, neighbors.cols, DataType<float>::type);
        int found = index.radiusSearch(p, n, dist, radius, neighbors.cols);
        cout << "found=" << found << endl;
        cout << "p=" << p << endl;
        cout << "n=" << n << endl;
        cout << "dist=" << dist << endl;
        for(int j=0; j<neighbors.cols && j<found; ++j)
        {
            int ind = n.at<int>(0,j);
            assert(ind>=0);
            Mat pt2 = p2.row(ind);
            double dst = abs(p.at<float>(0,0)-pt2.at<float>(0,0))+abs(p.at<float>(0,1)-pt2.at<float>(0,1));
            cout << "dst=" << dst << ",";
            cout << "neighbor=[" << pt2.at<float>(0,0) << "," << pt2.at<float>(0,1) << ";" << endl;
            if (0 && dst>radius)
            {
                cout << "radius violation: query=" << p 
                     << "; neighbor=" << pt2
                     << "; L1=" << dst
                     << "; dist=" << dist.at<float>(0,j) << endl;
                cout << "j=" << j << endl;
                cout << "ind=" << ind << endl;
                cout << "found=" << found << endl;
                cout << "n=" << n << endl;
                cout << "dist=" << dist << endl;
            }
        }
        cout << endl;
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

class HarrisBinnedFeatureDetector : public cv::FeatureDetector
{
public:
    // descriptor radius is used only to init KeyPoints
    HarrisBinnedFeatureDetector(int radius, int n, int nbinx=24, int nbiny=5,
                                float k = .04, int block_size=3, int aperture_size=5)
        : m_radius(radius), m_nbinx(nbinx), m_nbiny(nbiny), 
          m_block_size(block_size), m_aperture_size(aperture_size), m_n(n)
        {
            assert(nbinx>0 && nbiny>0);
        }

protected:

    void
    detectImpl(cv::InputArray image, KeyPoints& kp, cv::InputArray mask=Mat()) const
    {
        Mat harris_response;
        // M_c = det(A) - k*trace^2(A), the range for k \in [0.04, 0.15]
        cv::cornerHarris(image, harris_response, m_block_size, m_aperture_size,
                         m_k, cv::BORDER_DEFAULT);
        assert(harris_response.type() == DataType<float>::type);
        int stridex = (int)image.getMat().cols/m_nbinx,
            stridey = (int)image.getMat().rows/m_nbiny;
        assert(stridex>0 && stridey>0);
        struct elem
        {
            int x, y;
            float val;
            elem(int x, int y, float val) : x(x), y(y), val(val) {}
            elem() : x(-1), y(-1), val(NAN) {}
            bool operator<(const elem& other) const{ return val<other.val; }
        };
        int corners_per_block = (int)m_n/(m_nbinx*m_nbiny);
        vector<elem> v;
        v.reserve(stridex*stridey);
        for(int binx=0; binx<m_nbinx; ++binx)
        {
            for(int biny=0; biny<m_nbiny; ++biny)
            {
                for(int x=binx*stridex; x<(binx+1)*stridex && x<harris_response.cols; ++x)
                {
                    for(int y=biny*stridey; y<(biny+1)*stridey && y<harris_response.rows; ++y)
                    {
                        float response = abs(harris_response.at<float>(y,x));
                        if (isEqual(response,.0f))
                            continue;
                        v.push_back(elem(x,y,response));
                    }
                }
                int m = (v.size()>corners_per_block) ? v.size()-corners_per_block : 0;
                if (m>0)
                    std::nth_element(v.begin(), v.begin()+m, v.end());
                for(vector<elem>::iterator iter=v.begin()+m; iter<v.end(); ++iter)
                {
                    KeyPoint keypoint;
                    keypoint.pt = Point2f(iter->x, iter->y);
                    keypoint.response = iter->val;
                    keypoint.size = 2*m_radius+1;
                    kp.push_back(keypoint);
                }
                v.clear();
            }
        }
        BOOST_LOG_TRIVIAL(info) << "found " << kp.size() << " harris corners";
    }
    int m_radius, m_nbinx, m_nbiny, m_block_size, m_aperture_size,m_n;
    float m_k;
};

class MyFeatureExtractor : public cv::DescriptorExtractor
{
public:
    MyFeatureExtractor(int descriptor_radius) 
        : m_descriptor_radius(descriptor_radius) {}
protected:
    int m_descriptor_radius;

    int defaultNorm() const
    {
        return cv::NORM_L1;
    }
    int descriptorType() const
    {
        return DataType<float>::type;
    }

    int 
    descriptorSize() const
    {
        return (2*m_descriptor_radius+1)*(2*m_descriptor_radius+1);
    }

    void computeImpl(InputArray image, std::vector<KeyPoint>& kp, OutputArray d) const
    {
        Mat sob(image.rows(), image.cols(), DataType<float>::type, Scalar(0));

        Mat(kp.size(), descriptorSize(), DataType<float>::type, Scalar(0)).copyTo(d);
        assert(d.getMat().data);
        Sobel(image, sob, sob.type(), 1, 0, 3, 1, 0, cv::BORDER_REFLECT_101);
        for(int k=0; k<kp.size(); ++k)
        {
            Point2i p = kp.at(k).pt;
            for(int i=-m_descriptor_radius,col=0; i<=m_descriptor_radius; i+=1)
            {
                for(int j=-m_descriptor_radius; j<=m_descriptor_radius; j+=1,++col)
                {
                    float val = (p.y+i>0 && p.y+i<image.rows() && p.x+j>0 && p.x+j<image.cols()) ?
                        sob.at<float>(p.y+i, p.x+j) : 0;
                    d.getMat().at<float>(k,col) = val;
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

void
mat2eig(const Mat& X, const Mat& X_prev, MatrixXf& Xe, MatrixXf& Xe_prev,
        Matches& match_pcl)
{
    assert(X.type() == DataType<float>::type);
    assert(X_prev.type() == DataType<float>::type);
    for(int i=0; i<match_pcl.size(); ++i)
    {
        int ind1 = match_pcl.at(i)[0], ind2 = match_pcl.at(i)[1];
        Xe(0,i) = X.at<float>(0,ind1);
        Xe(1,i) = X.at<float>(1,ind1);
        Xe(2,i) = X.at<float>(2,ind1);
        Xe_prev(0,i) = X_prev.at<float>(0,ind2);
        Xe_prev(1,i) = X_prev.at<float>(1,ind2);
        Xe_prev(2,i) = X_prev.at<float>(2,ind2);
    }
}

int nchoosek(int n, int k)
{
    int currentCombination[k];
    for (int i=0; i<k; i++)
        currentCombination[i]=i;
    currentCombination[k-1] = k-1-1; // fill initial combination is real first combination -1 for last number, as we will increase it in loop

    do
    {
        if (currentCombination[k-1] == (n-1) ) // if last number is just before overwhelm
        {
            int i = k-1-1;
            while (currentCombination[i] == (n-k+i))
                i--;
            
            currentCombination[i]++;
            
            for (int j=(i+1); j<k; j++)
                currentCombination[j] = currentCombination[i]+j-i;
        }
        else
            currentCombination[k-1]++;
        
        for (int i=0; i<k; i++)
            printf("%d ", currentCombination[i]);
        printf("\n");
        
    } while (! ((currentCombination[0] == (n-1-k+1)) && (currentCombination[k-1] == (n-1))) );
}

void
ransacRigidMotion(const MatrixXf& P1, const MatrixXf& P2,
                  const MatrixXf& Xe, const MatrixXf& Xe_prev,
                  Affine3f& Tb, vector<int>& inliers)
{
    int N=100, model_size=3, max_sup_size=0;
    double max_sup_rms=DBL_MAX;
    for(int k=0; k<N; ++k)
    {
        set<int> sample = randomsample(model_size, Xe.cols());
        Eigen::MatrixXf X1(3, model_size), X2(3, model_size);
        int i=0;
        for(auto j: sample)
        {
            X1.col(i) = Xe.col(j);
            X2.col(i) = Xe_prev.col(j);
            ++i;
        }
        Eigen::Affine3f T;
        solveRigidMotion(X1, X2, T);
        vector<int> inl;
        double sample_rms = getRMS(X1, X2, T, inl, DBL_MAX);
        double thresh = .1;
        inl.clear();
        double support_rms = getRMS(Xe, Xe_prev, T, inl, thresh);
        if (inl.size()>max_sup_size) 
        {
            max_sup_rms = support_rms;
            max_sup_size = inliers.size();
            Tb = T;
            inliers = inl;
        }
    }
    BOOST_LOG_TRIVIAL(info) <<"max support set size=" << max_sup_size << " out of " << Xe.cols() << " its RMS=" << max_sup_rms << endl;
}

// stereo odometry

vector<Affine3f>
sequenceOdometry(const Mat& P1, const Mat& P2, StereoImageGenerator& images,
                 const boost::filesystem::path& dbg_dir)
{
    int MAX_FEATURE_NUM = 5000;
    HarrisBinnedFeatureDetector detector(5, MAX_FEATURE_NUM);
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
    Matches match_lr, match_lr_prev;
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
            im1.copyTo(im1_prev); im2.copyTo(im2_prev);
            d1.copyTo(d1_prev); d2.copyTo(d2_prev);
            kp1_prev = kp1; kp2_prev = kp2;
            match_lr_prev = match_lr;
            X.copyTo(X_prev);
            kp1.clear(); kp2.clear();
            match_lr.clear();
        }
        im1 = (*stereo_pair).first;
        im2 = (*stereo_pair).second;
        assert(im1.data && im2.data);
        detector.detect(im1,kp1);
	detector.detect(im2,kp2);
	BOOST_LOG_TRIVIAL(info) << "using " << kp1.size() << " keypoints in the 1st image";
	BOOST_LOG_TRIVIAL(info) << "using " << kp2.size() << " keypoints in the 2nd image";
        extractor.compute(im1, kp1, d1);
	extractor.compute(im2, kp2, d2);
        save1(im1, kp1, (boost::format((dbg_dir/"harris_left_%03d.jpg").string()) % iter_num).str().c_str(), INT_MAX);
        save1(im2, kp2, (boost::format((dbg_dir/"harris_right_%03d.jpg").string()) % iter_num).str().c_str(), INT_MAX);
        match_desc(kp1, kp2, d1, d2, match_lr, MatchParams(F));
        BOOST_LOG_TRIVIAL(info) << cv::format("Done matching left vs right: %d matches", match_lr.size());
        save2blend(im1, im2, kp1, kp2, match_lr, (boost::format((dbg_dir/"lr_blend_%03d.jpg").string()) % iter_num).str().c_str());

	cv::Mat
            x1(2, match_lr.size(), DataType<float>::type),
            x2(2, match_lr.size(), DataType<float>::type);
        collect_matches(kp1,kp2,match_lr,x1,x2);
        // each col is a 3d pt
        X = triangulate_dlt(x1, x2, P1, P2);
        assert(X.type()==DataType<float>::type);
        save1reproj(im1,X,x1,P1,(boost::format((dbg_dir/"tri_l%03d.jpg").string()) % iter_num).str().c_str());
        save1reproj(im1,X,x2,P2,(boost::format((dbg_dir/"tri_r%03d.jpg").string()) % iter_num).str().c_str());
        if (first)
        {
            first = false;
            continue;
        }

        Matches match11;
        match_desc(kp1, kp1_prev, d1, d1_prev, match11);
        save2blend(im1, im1_prev, kp1, kp1_prev, match11,
                   (boost::format((dbg_dir/"ll%d.jpg").string())%iter_num).str().c_str(), INT_MAX);
        BOOST_LOG_TRIVIAL(info) << cv::format("Done matching left vs left_prev: %d matches", match11.size());

        Matches match22;
        match_desc(kp2, kp2_prev, d2, d2_prev, match22);
        save2blend(im2, im2_prev, kp2, kp2_prev, match22,
                   (boost::format((dbg_dir/"rr%d.jpg").string())%iter_num).str().c_str(), INT_MAX);
        BOOST_LOG_TRIVIAL(info) << cv::format("Done matching right vs right_prev: %d matches", match22.size());

        Matches match_pcl; 
        vector<Vec4i> circ_match;
        match_circle(match_lr, match_lr_prev, match11, match22, circ_match, match_pcl);
        if (circ_match.size() < 3)
        {
            BOOST_LOG_TRIVIAL(info) << "not enough matches in current circle: " << circ_match.size();
            poses.push_back(Affine3f::Identity());
            continue;
        } 
        BOOST_LOG_TRIVIAL(info) << circ_match.size() << " points in circular match";
        MatrixXf Xe(3, match_pcl.size()), Xe_prev(3, match_pcl.size());
        mat2eig(X, X_prev, Xe, Xe_prev, match_pcl);
        save4(im1, im1_prev, im2, im2_prev, kp1, kp1_prev, kp2, kp2_prev, circ_match,
              (boost::format((dbg_dir/"circ_match_%03d.jpg").string()) % iter_num).str().c_str());
        BOOST_LOG_TRIVIAL(info) << "solving rigid motion";
        Affine3f T;
        MatrixXf P1e, P2e; cv2eigen(P1,P1e); cv2eigen(P2,P2e);
        vector<int> inliers;
        ransacRigidMotion(P1e, P2e, Xe, Xe_prev, T, inliers);
        poses.push_back(T);
        MatrixXf Xe_prev_rot = h2e(T.matrix()*e2h(Xe_prev));
        save2reproj(im1, get_inl(Xe,inliers), get_inl(Xe_prev_rot,inliers), P1e,
                    (boost::format((dbg_dir/"reproj_%03d.jpg").string()) % iter_num).str().c_str());
    }
    BOOST_LOG_TRIVIAL(info) << "avg time per iteration [s]:" << float(clock()-begin_time)/CLOCKS_PER_SEC/iter_num << endl;
    return poses;
}

void
calibratedSFM(const Mat& K, MonoImageGenerator& images)
{
    int MAX_FEATURE_NUM = 1500;
    HarrisBinnedFeatureDetector detector(9, MAX_FEATURE_NUM);
    MyFeatureExtractor extractor(9);
    MonoImageGenerator::result_type image;
    Mat im1, im1_prev;
    Mat d1, d1_prev;
    KeyPoints kp1, kp1_prev;
    Matches match, match_prev;
    const clock_t begin_time = clock();
    int iter_num;
    assert(K.type()==DataType<double>::type);
    float focal = K.at<double>(0,0);
    Point2f pp(K.at<double>(0,3), K.at<double>(1,3));
    for(iter_num=0; image=images(); ++iter_num)
    {
        BOOST_LOG_TRIVIAL(info) << "iter: " << iter_num;
        if (iter_num) 
        {
            im1.copyTo(im1_prev); d1.copyTo(d1_prev);
            kp1_prev = kp1; match_prev = match;
            kp1.clear(); match.clear();
        }
        im1 = *image; assert(im1.data);
        detector.detect(im1,kp1);
        assert(kp1.size()>0);
	BOOST_LOG_TRIVIAL(info) << "using " << kp1.size() << " keypoints in the 1st image";
        extractor.compute(im1, kp1, d1);
        save1(im1, kp1, (boost::format("kp_%03d.jpg") % iter_num).str().c_str(), INT_MAX);
        if (!iter_num)
            continue;
        MatchParams short_mp;
        short_mp.radius = 10;
        match_desc(kp1, kp1_prev, d1, d1_prev, match, short_mp);
        BOOST_LOG_TRIVIAL(info) << cv::format("Done matching: %d matches", match.size());
        save2blend(im1, im1, kp1, kp1_prev, match, (boost::format("match0_%03d.jpg") % iter_num).str().c_str());
	cv::Mat x1(2, match.size(), DataType<float>::type), x2(2, match.size(), DataType<float>::type);
        collect_matches(kp1, kp1_prev, match, x1, x2);
        Mat x1t, x2t;
        transpose(x1,x1t);
        transpose(x2,x2t);
        Mat x1t2c(x1t.rows, 1, CV_32FC2), x2t2c(x2t.rows, 1, CV_32FC2),
            x1t_normalized(x1t.rows, 1, CV_32FC2), x2t_normalized(x2t.rows, 1, CV_32FC2);
        for(int i=0; i<x1t.rows; ++i)
        {
            x1t2c.at<Vec2f>(i) = Vec2f(x1t.at<float>(i,0), x1t.at<float>(i,1));
            x2t2c.at<Vec2f>(i) = Vec2f(x2t.at<float>(i,0), x2t.at<float>(i,1));
        }
        undistortPoints(x1t2c, x1t_normalized, K, Mat());
        undistortPoints(x2t2c, x2t_normalized, K, Mat());
        Mat E = findEssentialMat(x1t, x2t, focal, pp);
        Mat F = (K.inv()).t()*E*(K.inv());
        MatchParams mp(F);
        mp.enforce_2nd_best = true;
        mp.ratio_2nd_best = .9;
        mp.radius = 10;
        match_desc(kp1, kp1_prev, d1, d1_prev, match,mp);
        save2blend(im1, im1_prev, kp1, kp1_prev, match,
                   (boost::format("match_%d.jpg")%iter_num).str().c_str(), INT_MAX);
        Mat P1(3,4,DataType<float>::type,Scalar(0)), P2(3,4,DataType<float>::type,Scalar(0));
        P1.at<float>(0,0) = P1.at<float>(1,1) = P1.at<float>(2,2) = 1.0;
        P2.at<float>(0,0) = P2.at<float>(1,1) = P2.at<float>(2,2) = 1.0;
    }
    BOOST_LOG_TRIVIAL(info) << "avg time per iteration [s]:" << float(clock()-begin_time)/CLOCKS_PER_SEC/iter_num << endl;
}
