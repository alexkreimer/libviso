#ifndef _VISO_H
#define _VISO_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <map>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iomanip>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include <boost/format.hpp>

#include <Eigen/Dense>

#include "mvg.h"
#include "misc.h"

using namespace std;
using namespace boost;

using cv::Mat;
using cv::KeyPoint;
using cv::Vec2i;
using cv::FeatureDetector;
using cv::DescriptorExtractor;
using cv::Scalar;
using cv::FileStorage;
using cv::Vec6f;
using cv::Point2i;
using cv::waitKey;
using cv::Size;
using cv::Point;
using cv::Point2f;
using cv::Vec3i;
using Eigen::MatrixXf;
using Eigen::Affine3f;

typedef vector<KeyPoint> KeyPoints;
typedef Vec3i Match; //i1, i2, dist
typedef vector<Match> Matches;
typedef vector<Point2f> Points2f;
typedef Mat Descriptors;
typedef pair<Mat,Mat> image_pair;

class StereoImageGenerator {
public:
    typedef boost::optional<image_pair> result_type;
    typedef pair<string, string> string_pair;
    StereoImageGenerator(const string_pair &mask,int begin=0, int end=INT_MAX)
	: m_mask(mask), m_index(begin), m_end(end) {}
    result_type operator()() {
        if (m_index > m_end)
            return result_type();
	string name0 = str(boost::format(m_mask.first) % m_index),
               name1 = str(boost::format(m_mask.second) % m_index);
        image_pair pair = make_pair(cv::imread(name0, CV_LOAD_IMAGE_GRAYSCALE),
                                    cv::imread(name1, CV_LOAD_IMAGE_GRAYSCALE));
        m_index++;
	return pair.first.data && pair.second.data ? result_type(pair) : result_type();
    }
private:
    int m_index, m_end;
    string_pair m_mask;
};

/*** Read the intrinsic and extrinsic parameters
     Note: this works for KITTI, will probably need to be updated for another dataset 
*/
void
readCameraParams(const string &intrinsics_name, const string &extrinsics_name,
                 StereoCam &p);

Mat
getFundamentalMat(const Mat& R1, const Mat& t1,
		  const Mat& R2, const Mat& t2,
		  const Mat& cameraMatrix);
void
findConstrainedCorrespondences(const Mat& F, const KeyPoints& kp1,
                               const KeyPoints& kp2,const Mat& d1,
			       const Mat& d2, Matches& matches,
			       double eps, double ratio);

vector<Affine3f>
sequenceOdometry(const Mat& p1, const Mat& p2, StereoImageGenerator& images);


void
match_l2_2nd_best(const Descriptors& d1, const Descriptors& d2,
                  Matches& match, float ratio=0.7);
void
collect_matches(const KeyPoints& kp1, const KeyPoints &kp2,
                const Matches &match, Points2f &p1, Points2f &p2, int lim=INT_MAX);

/* p2Fp1=0 */
void
match_epip_constraint(const cv::Mat& F, const KeyPoints& kp1, 
                      const KeyPoints& kp2, const Descriptors& d1,
                      const Descriptors& d2, Matches &match,
                      double ratio, double samp_thresh, double alg_thresh);
void
save2(const cv::Mat& m1, const cv::Mat& m2, const KeyPoints& kp1,
      const KeyPoints& kp2, const Matches &match, const string &file_name,
      int lim=50);
#endif
