#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>

#include "viso.h"

using boost::format;
using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::Affine3f;

namespace fs = boost::filesystem;
namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace keywords = boost::log::keywords;

bool
loadCalib(const string& file_name, Mat& p1, Mat& p2)
{
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp)
      return false;
  int n,t;
  t = fscanf(fp, "P%d:", &n);
  assert(t==1);
  t=fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
           &p1.at<double>(0,0), &p1.at<double>(0,1), &p1.at<double>(0,2), &p1.at<double>(0,3),
           &p1.at<double>(1,0), &p1.at<double>(1,1), &p1.at<double>(1,2), &p1.at<double>(1,3),
           &p1.at<double>(2,0), &p1.at<double>(2,1), &p1.at<double>(2,2), &p1.at<double>(2,3));
  assert(t==12);
  t=fscanf(fp, "P%d:", &n);
  assert(t==1);
  t=fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
           &p2.at<double>(0,0), &p2.at<double>(0,1), &p2.at<double>(0,2), &p2.at<double>(0,3),
           &p2.at<double>(1,0), &p2.at<double>(1,1), &p2.at<double>(1,2), &p2.at<double>(1,3),
           &p2.at<double>(2,0), &p2.at<double>(2,1), &p2.at<double>(2,2), &p2.at<double>(2,3));
  assert(t==12);
  fclose(fp);
  return true;
}


bool
savePoses(const string& file_name, const vector<Affine3f>& poses) {
    FILE *fp = fopen(file_name.c_str(),"w+");
    if (!fp)
        false;
    for (auto &pose: poses)
    {
        int t=fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                      pose(0,0), pose(0,1), pose(0,2), pose(0,3),
                      pose(1,0), pose(1,1), pose(1,2), pose(1,3),
                      pose(2,0), pose(2,1), pose(2,2), pose(2,3));
        assert(t>0);
    }
    fclose(fp);
    return true;
}

void init_log()
{
    logging::add_file_log
    (
        keywords::file_name = "kitti.log",
        keywords::rotation_size = 10 * 1024 * 1024,
        keywords::time_based_rotation = sinks::file::rotation_at_time_point(0, 0, 0),
        keywords::format = "[%TimeStamp%]: %Message%",
        keywords::auto_flush = true
    );
    logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::debug);
}

int main(int argc, char** argv)
{
    if (argc<3)
    {
        cout << "usage: kitti sha seq begin end ..." << endl;
        exit(1);
    }
//    init_log();
    char *result_sha = argv[1], *KITTI_HOME  = std::getenv("KITTI_HOME");
    assert(KITTI_HOME);
    fs::path seq_base = fs::path(KITTI_HOME) / "sequences";
    for(int j=2; argv[j]; ++j)
    {
        string seq_name(argv[j]);
        fs::path result_dir = fs::path(KITTI_HOME) / "results" / seq_name / result_sha;
        fs::create_directories(result_dir);
        BOOST_LOG_TRIVIAL(info) << "Processing sequence: " << seq_name;
        Mat P1(3,4,cv::DataType<double>::type), P2(3,4,cv::DataType<double>::type);
        string calib_file_name = (seq_base/seq_name/"calib.txt").string();
        BOOST_LOG_TRIVIAL(info) << "Read camera calibration info from: " << calib_file_name;
        bool res = loadCalib(calib_file_name, P1, P2);
        assert(res);
        StereoImageGenerator images(StereoImageGenerator::string_pair(
                                        (seq_base / seq_name / "image_0" / "%06d.png").string(),
                                        (seq_base / seq_name / "image_1" / "%06d.png").string()),0,300);
        vector<Affine3f> poses = sequenceOdometry(P1, P2, images, result_dir);
        vector<Affine3f> kitti_poses;
        Affine3f Tk = Affine3f::Identity();
        int i=0;
        for(auto &T: poses)
        {
            if (Tk.inverse().matrix().allFinite())
                kitti_poses.push_back(Tk.inverse());
            else {
                BOOST_LOG_TRIVIAL(info) << "estimation failed" << endl;
                kitti_poses.push_back(Affine3f::Identity());
            }
            Tk = T*Tk;
        }
        fs::path poses_dir(result_dir/"data");
        create_directories(poses_dir);
        string poses_file_name((poses_dir/(seq_name+".txt")).string());
        BOOST_LOG_TRIVIAL(info) << "Saving poses to " << poses_file_name;
        savePoses(poses_file_name, kitti_poses);
    }
    return 0;
}
