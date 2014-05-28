#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

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
#include "viso.h"

using namespace cv;
using namespace std;
using namespace boost;
using namespace log4cxx;
using namespace log4cxx::helpers;

LoggerPtr logger(Logger::getLogger("MyApp"));

LoggerPtr image_pair_generator::logger(Logger::getLogger("com.image_pair_generator"));

/*** Read the intrinsic and extrinsic parameters
     Note: this works for KITTI, will probably need to be updated for another dataset 
*/
void
read_camera_params(const string &intrinsics_name,
		   const string &extrinsics_name,
		   stereo_pair &p)
{
    LOG4CXX_DEBUG(logger, "Reading intrinsics...");
    FileStorage fs(intrinsics_name, FileStorage::READ);
    auto fn = [&](FileStorage fs) {
	FileNode root = fs.root();
	for(FileNodeIterator it = root.begin(); it != root.end(); ++it)
	{
	    FileNode node = *it;
	    if (node.isNamed()) {
		stringstream msg;
		msg << "node name: " << node.name();
		LOG4CXX_DEBUG(logger, msg.str());
	    } else {
		LOG4CXX_DEBUG(logger, (*it).type());
	    }
	}
    };

    if (!fs.isOpened()) {
	throw runtime_error(str(boost::format("Can not open intrinsic params file %s") % intrinsics_name));
    } else {
	fn(fs);
    }
    fn(fs);
    Mat t;
    fs["M1"] >> t;
    p.c1.K = t.clone();
    p.c1.K.convertTo(p.c1.K, CV_32FC1);
    fs["D1"] >> t;
    p.c1.D = t.clone();
    fs["M2"] >> t;
    p.c2.K = t.clone();
    p.c2.K.convertTo(p.c2.K, CV_32FC1);
    fs["D2"] >> t;
    p.c2.D = t.clone();
    fs.release();
    LOG4CXX_DEBUG(logger, "Done reading extrinsics...");
    fs.open(extrinsics_name, FileStorage::READ);
    if (!fs.isOpened()) {
	throw runtime_error(str(boost::format("Can not open extrinsic params file %s") % extrinsics_name));
    } else {
	fn(fs);
    }
    fs["R"]  >> t;
    p.R = t.clone();
    fs["T"]  >> t;
    p.t = t.clone();
    fs["R1"] >> t;
    p.R1 = t.clone();
    fs["R2"] >> t;
    p.R2 = t.clone();
    fs["P1"] >> t;
    p.P1 = t.clone();
    fs["P2"] >> t;
    p.P2 = t.clone();
    fs["Q"]  >> t;
    p.Q = t.clone();
    LOG4CXX_DEBUG(logger, "done reading extrinsics...");
    fs.release();
}


bool readCameraMatrix(const string& filename,
                             Mat& cameraMatrix, Mat& distCoeffs,
                             Size& calibratedImageSize )
{
    FileStorage fs(filename, FileStorage::READ);
    fs["image_width"] >> calibratedImageSize.width;
    fs["image_height"] >> calibratedImageSize.height;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;

    if( distCoeffs.type() != CV_64F )
        distCoeffs = Mat_<double>(distCoeffs);
    if( cameraMatrix.type() != CV_64F )
        cameraMatrix = Mat_<double>(cameraMatrix);

    return true;
}

struct PointModel
{
    vector<Point3f> points;
    vector<vector<int> > didx;
    Mat descriptors;
    string name;
};


void 
writeModel(const string& modelFileName, const string& modelname,
                       const PointModel& model)
{
    FileStorage fs(modelFileName, FileStorage::WRITE);

    fs << modelname << "{" <<
        "points" << "[:" << model.points << "]" <<
        "idx" << "[:";

    for( size_t i = 0; i < model.didx.size(); i++ )
        fs << "[:" << model.didx[i] << "]";
    fs << "]" << "descriptors" << model.descriptors;
}


void unpackPose(const Vec6f& pose, Mat& R, Mat& t)
{
    Mat rvec = (Mat_<double>(3,1) << pose[0], pose[1], pose[2]);
    t = (Mat_<double>(3,1) << pose[3], pose[4], pose[5]);
    Rodrigues(rvec, R);
}


Mat getFundamentalMat( const Mat& R1, const Mat& t1,
                              const Mat& R2, const Mat& t2,
                              const Mat& cameraMatrix )
{
    Mat_<double> R = R2*R1.t(), t = t2 - R*t1;
    double tx = t.at<double>(0,0), ty = t.at<double>(1,0), tz = t.at<double>(2,0);
    Mat E = (Mat_<double>(3,3) << 0, -tz, ty, tz, 0, -tx, -ty, tx, 0)*R;
    Mat iK = cameraMatrix.inv();
    Mat F = iK.t()*E*iK;

#if 0
    static bool checked = false;
    if(!checked)
    {
        vector<Point3f> objpoints(100);
        Mat O(objpoints);
        randu(O, Scalar::all(-10), Scalar::all(10));
        vector<Point2f> imgpoints1, imgpoints2;
        projectPoints(Mat(objpoints), R1, t1, cameraMatrix, Mat(), imgpoints1);
        projectPoints(Mat(objpoints), R2, t2, cameraMatrix, Mat(), imgpoints2);
        double* f = (double*)F.data;
        for( size_t i = 0; i < objpoints.size(); i++ )
        {
            Point2f p1 = imgpoints1[i], p2 = imgpoints2[i];
            double diff = p2.x*(f[0]*p1.x + f[1]*p1.y + f[2]) +
		p2.y*(f[3]*p1.x + f[4]*p1.y + f[5]) +
                f[6]*p1.x + f[7]*p1.y + f[8];
            CV_Assert(fabs(diff) < 1e-3);
        }
        checked = true;
    }
#endif
    return F;
}


void
findConstrainedCorrespondences(const Mat& _F,
			       const vector<KeyPoint>& keypoints1,
			       const vector<KeyPoint>& keypoints2,
			       const Mat& descriptors1,
			       const Mat& descriptors2,
			       vector<Vec2i>& matches,
			       double eps, double ratio)
{
    float F[9]={0};
    int dsize = descriptors1.cols;

    Mat Fhdr = Mat(3, 3, CV_32F, F);
    _F.convertTo(Fhdr, CV_32F);
    matches.clear();

    for( int i = 0; i < (int)keypoints1.size(); i++ )
    {
        Point2f p1 = keypoints1[i].pt;
        double bestDist1 = DBL_MAX, bestDist2 = DBL_MAX;
        int bestIdx1 = -1;//, bestIdx2 = -1;
        const float* d1 = descriptors1.ptr<float>(i);

        for( int j = 0; j < (int)keypoints2.size(); j++ )
        {
            Point2f p2 = keypoints2[j].pt;
            double e = p2.x*(F[0]*p1.x + F[1]*p1.y + F[2]) +
		p2.y*(F[3]*p1.x + F[4]*p1.y + F[5]) +
		F[6]*p1.x + F[7]*p1.y + F[8];
            if( fabs(e) > eps )
                continue;
            const float* d2 = descriptors2.ptr<float>(j);
            double dist = 0;
            int k = 0;

            for( ; k <= dsize - 8; k += 8 )
            {
                float t0 = d1[k] - d2[k], t1 = d1[k+1] - d2[k+1];
                float t2 = d1[k+2] - d2[k+2], t3 = d1[k+3] - d2[k+3];
                float t4 = d1[k+4] - d2[k+4], t5 = d1[k+5] - d2[k+5];
                float t6 = d1[k+6] - d2[k+6], t7 = d1[k+7] - d2[k+7];
                dist += t0*t0 + t1*t1 + t2*t2 + t3*t3 +
		    t4*t4 + t5*t5 + t6*t6 + t7*t7;

                if( dist >= bestDist2 )
                    break;
            }

            if( dist < bestDist2 )
            {
                for( ; k < dsize; k++ )
                {
                    float t = d1[k] - d2[k];
                    dist += t*t;
                }

                if( dist < bestDist1 )
                {
                    bestDist2 = bestDist1;
                    //bestIdx2 = bestIdx1;
                    bestDist1 = dist;
                    bestIdx1 = (int)j;
                }
                else if( dist < bestDist2 )
                {
                    bestDist2 = dist;
                    //bestIdx2 = (int)j;
                }
            }
        }

        if( bestIdx1 >= 0 && bestDist1 < bestDist2*ratio )
        {
            Point2f p2 = keypoints1[bestIdx1].pt;
            double e = p2.x*(F[0]*p1.x + F[1]*p1.y + F[2]) +
		p2.y*(F[3]*p1.x + F[4]*p1.y + F[5]) +
		F[6]*p1.x + F[7]*p1.y + F[8];
            if( e > eps*0.25 )
                continue;
            double threshold = bestDist1/ratio;
            const float* d22 = descriptors2.ptr<float>(bestIdx1);
            int i1 = 0;
            for( ; i1 < (int)keypoints1.size(); i1++ )
            {
                if( i1 == i )
                    continue;
                Point2f pt1 = keypoints1[i1].pt;
                const float* d11 = descriptors1.ptr<float>(i1);
                double dist = 0;

                e = p2.x*(F[0]*pt1.x + F[1]*pt1.y + F[2]) +
                    p2.y*(F[3]*pt1.x + F[4]*pt1.y + F[5]) +
                    F[6]*pt1.x + F[7]*pt1.y + F[8];
                if( fabs(e) > eps )
                    continue;

                for( int k = 0; k < dsize; k++ )
                {
                    float t = d11[k] - d22[k];
                    dist += t*t;
                    if( dist >= threshold )
                        break;
                }

                if( dist < threshold )
                    break;
            }
            if( i1 == (int)keypoints1.size() )
                matches.push_back(Vec2i(i,bestIdx1));
        }
    }
}

typedef pair<int, int> Pair2i;
typedef map<Pair2i, int> Set2i;

struct EqKeypoints
{
    EqKeypoints(const vector<int>* _dstart, const Set2i* _pairs)
	: dstart(_dstart), pairs(_pairs) {}

    bool operator()(const Pair2i& a, const Pair2i& b) const
	{
	    return pairs->find(Pair2i(dstart->at(a.first) + a.second,
				      dstart->at(b.first) + b.second)) != pairs->end();
	}

    const vector<int>* dstart;
    const Set2i* pairs;
};

void
build3dmodel(const FeatureDetector& detector,
	     const DescriptorExtractor& extractor,
	     const stereo_pair& p,
	     image_pair_generator& all_images)
{
    Set2i pairs, keypointsIdxMap;
    Mat F = p.fund();
    image_pair_generator::result_type pair;
    while(pair = all_images())
    {
	LOG4CXX_DEBUG(logger, "Detecting keypoints...");
        Mat im1 = (*pair).first, im2 = (*pair).second;
        vector<KeyPoint> keypoints1, keypoints2;
        detector.detect(im1, keypoints1);
	detector.detect(im2, keypoints2);
	LOG4CXX_DEBUG(logger, "Computing descriptors...");
	Mat descriptors1, descriptors2;
        extractor.compute(im1, keypoints1, descriptors1);
	extractor.compute(im2, keypoints2, descriptors2);
	LOG4CXX_DEBUG(logger, "Computing matches...");
	int pairsFound = 0;
	vector<Vec2i> match;
	const double Feps = 5;
	const double DescriptorRatio = 0.7;
	LOG4CXX_DEBUG(logger, "Computing fundamental constrained correspondences...");
	findConstrainedCorrespondences(F, keypoints1, keypoints2, descriptors1,
				       descriptors2, match, Feps, DescriptorRatio);
	vector<Point2f> pts1, pts2;
	for(size_t k=0; k<match.size(); k++)
	{
	    int i1 = match[k][0], i2 = match[k][1];
	    pts1.push_back(keypoints1[i1].pt);
	    pts2.push_back(keypoints2[i2].pt);
	}
	LOG4CXX_DEBUG(logger, "3d reconstruction...");
	vector<Point3f> objpts = p.triangulate(pts1, pts2);
    }
}
