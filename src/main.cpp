#include "viso.h"

int main(int argc, char** argv)
{
    BasicConfigurator::configure();
    LOG4CXX_DEBUG(logger, "Entering application.");
    string intrinsics_filename("/home/kreimer/sahar/viso/data/intrinsics.yml"), 
	extrinsics_filename("/home/kreimer/sahar/viso/data/extrinsics.yml");
    stereo_pair p;
    read_camera_params(intrinsics_filename, extrinsics_filename, p);
    image_pair_generator all_images(image_pair_generator::string_pair("../00/image_0/%06d.png", "../00/image_1/%06d.png"));
    image_pair_generator::result_type pair;
    SurfFeatureDetector detector(400);
    SurfDescriptorExtractor extractor;
    build3dmodel(detector, extractor, p, all_images);
    LOG4CXX_INFO(logger, "Exiting application.")
    return 0;
}
