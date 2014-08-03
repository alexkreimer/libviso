#include <boost/format.hpp>
#include <boost/log/trivial.hpp>

#include "viso.h"

using boost::format;
using cv::Mat;

extern bool test_P_from_KRt();
int main(int argc, char** argv)
{
    string image1_file_name = "WP_000401.jpg", image2_file_name = "WP_000404.jpg";
    Mat im1 = cv::imread(image1_file_name, 0), im2 = cv::imread(image2_file_name, 0);
    BOOST_LOG_TRIVIAL(debug) << "done reading images";
    cv::SiftFeatureDetector detector;//0, 5, 0.04, 5);
    cv::SiftDescriptorExtractor extractor;
    KeyPoints kp1, kp2;
    Mat mask1 = Mat::zeros(im1.size(), CV_8U), roi1(mask1, cv::Rect(110, 923, 2364, 736)),
        mask2 = Mat::zeros(im2.size(), CV_8U), roi2(mask2, cv::Rect(218, 907, 2124, 708));
    roi1 = Scalar(255, 255, 255);
    roi2 = Scalar(255, 255, 255);
    detector.detect(im1, kp1, mask1);
    detector.detect(im2, kp2, mask2);
    BOOST_LOG_TRIVIAL(debug) << cv::format("done computing interest points: %d (%d) keypoints found in the first (second) image", kp1.size(), kp2.size());
    Descriptors d1, d2;
    extractor.compute(im1, kp1, d1);
    extractor.compute(im2, kp2, d2);
    BOOST_LOG_TRIVIAL(debug) << "done computing descriptors";
    Matches match;
    match_l2_2nd_best(d1, d2, match);
    BOOST_LOG_TRIVIAL(debug) << "done with 2nd best match";
    save2(im1, im2, kp1, kp2, match, "match.jpg");
    Points2f p1, p2;
    collect_matches(kp1, kp2, match, p1, p2);
    // p2'Fp1 = 0
    Mat F = findFundamentalMat(p1, p2, cv::FM_RANSAC, 3, 0.99);
    assert(F.type() == CV_64F);
    BOOST_LOG_TRIVIAL(debug) << "done computing fundamental" + _str<double>(F);
    match_epip_constraint(F, kp1, kp2, d1, d2, match, .7, 2, .5);
    BOOST_LOG_TRIVIAL(debug) << (boost::format("done matching with epipolar constraint: %d matches") % match.size()).str().c_str();
    collect_matches(kp1, kp2, match, p1, p2);
    Mat H = findHomography(p1, p2, CV_RANSAC);
    BOOST_LOG_TRIVIAL(debug) <<  "done computing homography";
    Mat im1_warped;
    warpPerspective(im1, im1_warped, H, im1.size());
    BOOST_LOG_TRIVIAL(debug) << "done warping image";
    cv::imwrite("warped.jpg", im1_warped);
    return 0;
}
