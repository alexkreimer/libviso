libviso
=======

Visual Odometry library


StereoScan circular match

The goal of this project is to develop circular match procedure as described in the original StereoScan algorithm http://www.cvlibs.net/publications/Geiger2011IV.pdf

Things that you need to implement (feel free to add questions here, I will do my best to clarify things as we go on):

0. Take a look at the code in this repo.  It does many things that relevant to you.
1. Filter the image with the specified corner and blob detector masks as done in the paper
2. Do NMS.  Suggest the implementation based on the linked paper.
3. Compute Sobel responses and compute the descriptors as specified in the paper
4. Implement circular match.
5. Read and understand what they use triangulation for.  I did not go into the details here, so I would be glad if you explain it to me.
6. Implement hierarchical matching based on larger area NMS for speed up.


In order to start working you need:
* Ubuntu (basically anything about 12.04 will do, but go with 14.04 if you have an option).  You need either to install it on your computer by repartitioning it or to install virual machine (use VMWare player or VirtualBox for this. WARNING: this may be slow).
* OpenCV, download and install 3.0
* Eigen (just install the latest)
* see top level CMakeLists.txt, there are paths to the libs that you will need to change
* clone the repository to your computer and build it (if this succeedes you are in good shape): 
git clone https://github.com/alexkreimer/libviso.git
git checkout ss_circular_match
cd ss_circular_match
mkdir debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make


Good luck
