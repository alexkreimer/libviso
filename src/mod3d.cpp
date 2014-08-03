#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>

#include "viso.h"

using boost::format;
using Eigen::Affine3f;

namespace fs = boost::filesystem;
namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace keywords = boost::log::keywords;

bool
loadIntrinsics(const string& file_name, Mat& K)
{
    FILE *fp = fopen(file_name.c_str(),"r");
    if (!fp)
        return false;
    int n;
    assert(12==fscanf(fp, "K: %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                      &K.at<double>(0,0), &K.at<double>(0,1), &K.at<double>(0,2), &K.at<double>(0,3),
                      &K.at<double>(1,0), &K.at<double>(1,1), &K.at<double>(1,2), &K.at<double>(1,3),
                      &K.at<double>(2,0), &K.at<double>(2,1), &K.at<double>(2,2), &K.at<double>(2,3)));
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

int
main(int argc, char** argv)
{
    if (argc != 3)
    {
        cout << "usage: mod3d <dir> <seq_pattern>" << endl;
        cout << "for example: mod3d /home/kreimer/data/seq0 image_%03d.jpg" << endl;
        exit(1);
    }
    fs::path base_dir(argv[1]);
    string intrinsics_file_name = (base_dir/"intrinsics.txt").string();
    BOOST_LOG_TRIVIAL(trace) << "Load camera intrinsics from: " << intrinsics_file_name;
    Mat K(3,4,cv::DataType<double>::type);
    assert(loadIntrinsics(intrinsics_file_name, K));
    for(int i=0; i<INT_MAX; ++i)
    {
    }
    return 0;
}
