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
savePoses(const string& file_name, const vector<Mat>& poses) {
    FILE *fp = fopen(file_name.c_str(),"w+");
    if (!fp)
        false;
    for (auto &pose: poses)
    {
        int t=fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                      pose.at<double>(0,0), pose.at<double>(0,1), pose.at<double>(0,2), pose.at<double>(0,3),
                      pose.at<double>(1,0), pose.at<double>(1,1), pose.at<double>(1,2), pose.at<double>(1,3),
                      pose.at<double>(2,0), pose.at<double>(2,1), pose.at<double>(2,2), pose.at<double>(2,3));
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

extern int search_radius;

int main(int argc, char** argv)
{
    if (argc<4)
    {
        cout << "usage: demo result_sha seq_name search_radius begin end" << endl;
        exit(1);
    }
    int begin = 0, end = INT_MAX;
    search_radius = std::stoi(argv[3]);
    cout << "search_radius:" << search_radius << endl;
    if (argc > 4)
    {
        begin = std::stoi(argv[4]);
    }
    if (argc > 5)
    {
        end = std::stoi(argv[5]);
    }
    char *result_sha = argv[1], *KITTI_HOME  = std::getenv("KITTI_HOME");
    assert(KITTI_HOME);
    fs::path seq_base = fs::path(KITTI_HOME) / "sequences";
    string seq_name(argv[2]);
    fs::path result_dir = fs::path(KITTI_HOME) / ".." / "results" / result_sha / "data";
    fs::path debug_dir = fs::path(KITTI_HOME) / ".." / "results" / result_sha /"debug";

    fs::create_directories(result_dir);
    fs::create_directories(debug_dir);

    BOOST_LOG_TRIVIAL(info) << "Processing sequence: " << seq_name;
    Mat P1(3,4,cv::DataType<double>::type), P2(3,4,cv::DataType<double>::type);
    string calib_file_name = (seq_base/seq_name/"calib.txt").string();
    BOOST_LOG_TRIVIAL(info) << "Read camera calibration info from: " << calib_file_name;
    bool res = loadCalib(calib_file_name, P1, P2);
    assert(res);
    StereoImageGenerator images(StereoImageGenerator::string_pair(
                                    (seq_base / seq_name / "image_0" / "%06d.png").string(),
                                    (seq_base / seq_name / "image_1" / "%06d.png").string()),begin,end);
    vector<Mat> poses = sequence_odometry(P1, P2, images, debug_dir);
    string poses_file_name((result_dir/(seq_name+".txt")).string());
    BOOST_LOG_TRIVIAL(info) << "Saving poses to " << poses_file_name;
    savePoses(poses_file_name, poses);
    return 0;
}
