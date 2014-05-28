#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <map>
#include <stdexcept>
#include <vector>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include <boost/format.hpp>

#include "log4cxx/logger.h"
#include "log4cxx/basicconfigurator.h"
#include "log4cxx/helpers/exception.h"

#include "triangulation.h"

using namespace cv;
using namespace std;
using namespace boost;
using namespace log4cxx;
using namespace log4cxx::helpers;

extern LoggerPtr logger;

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
P_from_KRt(const Mat &K, const Mat &R, const Vec3f &t);

class pinhole_camera {
public:
    // intrinics
    Mat K;
    // distortion params
    Vec4f D;
};

class stereo_pair {
public:
    pinhole_camera c1,c2;
    /* rotation c1 -> c2 */
    Mat R;
    Vec3f t;
    Mat p1() const {
	Vec3f z;
	return P_from_KRt(c1.K, Mat::eye(3, 3, CV_32F), z);
    }
    Mat p2() const {
	return P_from_KRt(c2.K, R, t);
    }
    Mat fund() const {
	return F_from_P(p1(),p2());
    }

    /*** currently this does DLT, but will turn to be geometric error minimization */
    vector<Point3f>
    triangulate(const vector<Point2f>& x1, const vector<Point2f>& x2) const {
	return triangulate_dlt(x1, x2, p1(), p2());
    }
    /* rectification information */
    /* rotation of c1/c2 to get a rectified pair */
    Mat R1, R2;
    /* rectified camera matrices */
    Mat P1, P2;
    Mat Q;
};

typedef pair<Mat,Mat> image_pair;

class image_pair_generator {
public:
    static log4cxx::LoggerPtr logger;
    typedef boost::optional<image_pair> result_type;
    typedef pair<string, string> string_pair;
    image_pair_generator(const string_pair &mask,int index_begin=0)
	: m_mask(mask), m_index(index_begin) {}
    result_type operator()() {
	string name0 = str(boost::format(m_mask.first) % m_index),
	       name1 = str(boost::format(m_mask.second) % m_index);
	LOG4CXX_DEBUG(logger, str(boost::format("stereo pair %s, %s") % name0 % name1))
	image_pair pair = make_pair(imread_gray(name0), imread_gray(name1));
	return pair.first.data && pair.second.data ? result_type(pair) : result_type();
    }
private:
    // for some reason, this is said to produce a better results than directly reading with imread
    // http://stackoverflow.com/questions/7461075/opencv-image-conversion-from-rgb-to-grayscale-using-imread-giving-poor-results
    Mat imread_gray(const string &file) const {
	Mat image = imread(file), image_gray;
	cvtColor(image, image_gray, CV_BGR2GRAY);
	return image;
    }
    int m_index;
    string_pair m_mask;
};

/*** Read the intrinsic and extrinsic parameters
     Note: this works for KITTI, will probably need to be updated for another dataset 
*/
void
read_camera_params(const string &intrinsics_name,
		   const string &extrinsics_name,
		   stereo_pair &p);

Mat
getFundamentalMat(const Mat& R1, const Mat& t1,
		  const Mat& R2, const Mat& t2,
		  const Mat& cameraMatrix);
void
findConstrainedCorrespondences(const Mat& _F,
			       const vector<KeyPoint>& keypoints1,
			       const vector<KeyPoint>& keypoints2,
			       const Mat& descriptors1,
			       const Mat& descriptors2,
			       vector<Vec2i>& matches,
			       double eps, double ratio);
void build3dmodel(const FeatureDetector& detector,
		  const DescriptorExtractor& descriptor,
		  const stereo_pair& p,
		  image_pair_generator& all_images);
