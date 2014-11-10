libviso: Visual Odometry library
================================

You will need to download kitti dataset if you would like to test the code: here: http://www.cvlibs.net/download.php?file=data_odometry_gray.zip

Edit src/CMakeLists.txt and set opencv and eigen path's. You can download eigen here http://eigen.tuxfamily.org/.  

Build the code:

mkdir debug
cd debug
cmake ..
make


After you do this you will have debug/src/kitti executable.  set KITTI_HOME env variable to where you put your kitti data and now you can run the exec:
* result_sha is a run name (your choose it)
* seq_name is the name of kitti sequence (i.e., 00)
* begin end are first and last frame that will be processed