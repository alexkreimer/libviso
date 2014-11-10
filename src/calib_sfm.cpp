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
loadIntrinsics(const string& file_name, Mat& p1)
{
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp)
      return false;
  int n,t;
  fscanf(fp, "K:");
  t=fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
           &p1.at<double>(0,0), &p1.at<double>(0,1), &p1.at<double>(0,2),
           &p1.at<double>(1,0), &p1.at<double>(1,1), &p1.at<double>(1,2),
           &p1.at<double>(2,0), &p1.at<double>(2,1), &p1.at<double>(2,2));
  assert(t==9);
  fclose(fp);
  return true;
}
#if 0
void init_log()
{
    logging::add_file_log
    (
        keywords::file_name = "sfm.log",
        keywords::rotation_size = 10 * 1024 * 1024,
        keywords::time_based_rotation = sinks::file::rotation_at_time_point(0, 0, 0),
        keywords::format = "[%TimeStamp%]: %Message%",
        keywords::auto_flush = true
    );
    logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::debug);
}
#endif
int
main(int argc, char** argv)
{
    char *CBT_HOME = std::getenv("CBT_HOME");
    assert(CBT_HOME);
    fs::path result_dir = fs::path(CBT_HOME) / "results", seq_base = fs::path(CBT_HOME);
    Mat K(3,3,cv::DataType<double>::type);
    string calib_file_name = (seq_base/"calib.txt").string();
    BOOST_LOG_TRIVIAL(info) << "Read camera calibration info from: " << calib_file_name;
    bool res = loadIntrinsics(calib_file_name, K);
    assert(res);
    MonoImageGenerator images((seq_base/"img-%04d.jpg").string(), 1, 500);
    calibratedSFM(K, images);
    return 0;
}
