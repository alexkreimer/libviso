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
  int n;
  assert(fscanf(fp, "P%d:", &n) == 1);
  assert(12==fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                    &p1.at<double>(0,0), &p1.at<double>(0,1), &p1.at<double>(0,2), &p1.at<double>(0,3),
                    &p1.at<double>(1,0), &p1.at<double>(1,1), &p1.at<double>(1,2), &p1.at<double>(1,3),
                    &p1.at<double>(2,0), &p1.at<double>(2,1), &p1.at<double>(2,2), &p1.at<double>(2,3)));
  assert(fscanf(fp, "P%d:", &n) == 1);
  assert(12==fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &p2.at<double>(0,0), &p2.at<double>(0,1), &p2.at<double>(0,2), &p2.at<double>(0,3),
                    &p2.at<double>(1,0), &p2.at<double>(1,1), &p2.at<double>(1,2), &p2.at<double>(1,3),
                    &p2.at<double>(2,0), &p2.at<double>(2,1), &p2.at<double>(2,2), &p2.at<double>(2,3)));
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
        assert(fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                       pose(0,0), pose(0,1), pose(0,2), pose(0,3),
                       pose(1,0), pose(1,1), pose(1,2), pose(1,3),
                       pose(2,0), pose(2,1), pose(2,2), pose(2,3))>0);
    }
    fclose(fp);
    return true;
}

void init_log()
{
    logging::add_file_log("kitti.log");
    logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::info);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "usage: demo result_sha" << endl;
        exit(1);
    }
    
    char *result_sha = argv[1], *KITTI_HOME  = std::getenv("KITTI_HOME");
    char *KITTI_HOME_DEFAULT="./";
    if (!KITTI_HOME)
        KITTI_HOME=KITTI_HOME_DEFAULT;
    fs::path result_dir = fs::path(KITTI_HOME) / "results" / result_sha,
        seq_base = fs::path(KITTI_HOME) / "sequences";
    vector<string> seq_names = {"00"};
    for(auto &seq_name: seq_names)
    {
        Mat P1(3,4,cv::DataType<double>::type), P2(3,4,cv::DataType<double>::type);
        string calib_file_name = (seq_base / 
                                  seq_name / 
                                  "calib.txt").string();
        BOOST_LOG_TRIVIAL(info) << "Read camera calibration info from: " << calib_file_name;
        assert(loadCalib(calib_file_name, P1, P2));
        StereoImageGenerator images(StereoImageGenerator::string_pair(
                                        (seq_base / seq_name / "image_0" / "%06d.png").string(),
                                        (seq_base / seq_name / "image_1" / "%06d.png").string()));
        vector<Affine3f> poses = sequenceOdometry(P1, P2, images, 1000);
        vector<Affine3f> kitti_poses;
        Affine3f Tk = Affine3f::Identity();
        for(auto &T: poses)
        {
            kitti_poses.push_back(Tk.inverse());
            Tk = T*Tk;
        }
        string poses_file_name((result_dir / (seq_name + ".txt")).string());
        savePoses(poses_file_name, kitti_poses);
        BOOST_LOG_TRIVIAL(info) << "Saved poses to " << poses_file_name;
    }
    return 0;
}
