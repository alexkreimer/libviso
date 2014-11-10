#define BOOST_TEST_MODULE viso_tests
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "../src/viso.h"
#include "../src/estimation.h"
#include "../src/mvg.h"

#if 0
BOOST_AUTO_TEST_CASE(test_triangulate_dlt)
{
    Mat P1 = (Mat_<double>(3,4) << 
              7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
              0.000000000000e+00, 0.000000000000e+00, 7.188560000000e+02,
              1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00,
              0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00);
    Mat P2 = (Mat_<double>(3,4) <<
              7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 
              -3.861448000000e+02, 0.000000000000e+00, 7.188560000000e+02,
              1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00,
              0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00);
    //Mat F = F_from_P<float>(P1, P2);
    //cout << "P1=" << _str<double>(P1) << endl;
    //cout << "P2=" << _str<double>(P2) << endl;
    
    int N = 1000;
    Mat X = (Mat_<float>(4,1) << 0,0,1,1);
    //cout << "X=" << _str<float>(X) << endl;
    // project X
    Mat x1h = P1*X;
    //cout << "x1h:" << endl << x1h << endl;
    Mat x1 = h2e<float>(x1h), x2 = h2e<float>(P2*X);
    //cout << "x1=" << _str<float>(x1) << endl;
    //cout << "x2=" << _str<float>(x2) << endl;
    Mat Xt = triangulate_dlt(x1,x2,P1,P2);
    //cout << "Xt=" << _str<float>(Xt) << endl;
//    BOOST_CHECK_SMALL(norm(h2e<float>(X)-Xt), 1e-2);
    BOOST_CHECK_SMALL(.0d, 1e-2);

}

template<typename T>
ostream& operator<< (ostream& out, const vector<T> v) {
    int last = v.size() - 1;
    out << "[";
    for(int i = 0; i < last; i++)
        out << v[i] << ", ";
    out << v[last] << "]";
    return out;
}

BOOST_AUTO_TEST_CASE(test_nl_rigid_motion)
{
    using cv::Mat;
    using cv::RNG;
    using namespace std;
    Mat P1 = (Mat_<float>(3,4) << 
              7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
              0.000000000000e+00, 0.000000000000e+00, 7.188560000000e+02,
              1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00,
              0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00);
    Mat P2 = (Mat_<float>(3,4) <<
              7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 
              -3.861448000000e+02, 0.000000000000e+00, 7.188560000000e+02,
              1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00,
              0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00);
    struct param p;
    p.base = abs(P2.at<float>(0,3));
    p.calib.f = P2.at<float>(0,0);
    p.calib.cu = P2.at<float>(0,2);
    p.calib.cv = P2.at<float>(1,2);
    cout << "stereo rig params: base=" << p.base << ", focal=" << p.calib.f 
         << ", principal point=(" <<p.calib.cu << "," << p.calib.cv << ");"
         << endl;
    int num_pts = 10, a=0, b=1000;
    cout << "number of points: " << num_pts << endl;
    Mat X(3,num_pts,cv::DataType<float>::type), Xtr(3,num_pts,cv::DataType<float>::type);
    RNG rng;
    for (int i=0; i<num_pts; ++i)
    {
        X.at<float>(0,i) = rng.uniform(a,b);
        X.at<float>(1,i) = rng.uniform(a,b);
        X.at<float>(2,i) = rng.uniform(a,b);
    }
    vector<double> tr0(6,0.0);
    tr0[3] = 1;
    cout << "transformation vector: " << tr0 << endl;
    Mat Tr(4,4,cv::DataType<float>::type);
    tr2mat(tr0, Tr);
    cout << "transformation matrix: " << endl << Tr << endl;
    Xtr = h2e<float>(Tr*e2h<float>(X));
    Mat x1 = h2e<float>(P1*e2h<float>(Xtr)),
        x2 = h2e<float>(P2*e2h<float>(Xtr));
    Mat x(4,num_pts,cv::DataType<float>::type);
    for (int i=0; i<num_pts; ++i)
    {
        x.at<float>(0,i) = x1.at<float>(0,i);
        x.at<float>(1,i) = x1.at<float>(1,i);
        x.at<float>(2,i) = x2.at<float>(0,i);
        x.at<float>(3,i) = x2.at<float>(1,i);
    }
    vector<int> active(num_pts);
    vector<double> tr(6,0.0);
    for(int i=0;i<num_pts;++i) active[i] = i;
    minimize_reproj(X,x,tr,p,active);
    double err=.0f;
    cout << "tr:";
    for(int i=0;i<tr.size();++i)
    {
        cout << tr[i] << " ";
        err += abs(tr[i]-tr0[i]);
    }
    
    BOOST_CHECK_SMALL(err, 1e-4);
}
#endif
void read_data(Mat& X, Mat& observe)
{
    FILE *fp = fopen("/home/kreimer/data.csv", "r");
    if (!fp) {
        perror("fopen");
        exit(0);
    }
    int num_pts;
    int n = fscanf(fp,"%d\n",&num_pts);
    if (n!=1) {
        perror("fscanf");
        exit(0);
    }
    X.create(3,num_pts,cv::DataType<double>::type);
    observe.create(4,num_pts,cv::DataType<double>::type);
    printf("reading %d points\n",num_pts);
    for(int i=0;i<num_pts; ++i)
    {
        double nop;
        n = fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                   &nop, &nop, &nop, &nop,
                   &observe.at<double>(0,i), &observe.at<double>(1,i),
                   &observe.at<double>(2,i), &observe.at<double>(3,i),
                   &X.at<double>(0,i), &X.at<double>(1,i), &X.at<double>(2,i));
        if (n<8) {
            printf("cound't read all data points");
            exit(0);
        }
        if (0)
            printf("%f %f %f %f %f %f %f\n",
                   observe.at<double>(0,i), observe.at<double>(1,i),
                   observe.at<double>(2,i), observe.at<double>(3,i),
                   X.at<double>(0,i), X.at<double>(1,i), X.at<double>(2,i));
    }
}

BOOST_AUTO_TEST_CASE(test_nl_rigid_motion1)
{
    using cv::Mat;
    using cv::RNG;
    using namespace std;
    struct param p;
    p.base = .5707;
    p.calib.f = 645.24;
    p.calib.cu = 635.96;
    p.calib.cv = 194.13;
    Mat X,observe;
    read_data(X,observe);
    vector<double> tr(6,0.0);
    vector<int> inliers;
    BOOST_REQUIRE_EQUAL(ransac_minimize_reproj(X,observe,tr,inliers,p),true);
    printf("tr: %g %g %g %g %g %g\n",tr[0],tr[1],tr[2],tr[3],tr[4],tr[5]);
}

/*
BOOST_AUTO_TEST_CASE(test_solveRigidMotion)
{
    using Eigen::MatrixXf;
    using Eigen::Affine3f;
    using Eigen::Matrix3f;
    using Eigen::Vector4f;

    MatrixXf X1(4,3);
    X1 << 0,1,0, 1,2,3, 0,0,3, 1,1,1;
    std::cout << "X1:" << std::endl <<X1 << std::endl;
    Matrix3f R(Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f::UnitX()));
    Affine3f T(R), T1;
    T.translation() << 1,2,3;
    std::cout << "T.matrix():" << std::endl << T.matrix() << std::endl;
    MatrixXf X2 = T.matrix()*X1;
    std::cout << "X2:" << std::endl << X2 << std::endl;
    MatrixXf X1e(3,X2.cols()), X2e(3,X2.cols());
    for(int i=0; i<X2.cols(); ++i)
    {
        X2e(0,i) = X2(0,i)/X2(3,i);
        X2e(1,i) = X2(1,i)/X2(3,i);
        X2e(2,i) = X2(2,i)/X2(3,i);
        X1e(0,i) = X1(0,i)/X1(3,i);
        X1e(1,i) = X1(1,i)/X1(3,i);
        X1e(2,i) = X1(2,i)/X1(3,i);
    }
    std::cout << "X1e:" << std::endl << X1e << std::endl;
    std::cout << "X2e:" << std::endl << X2e << std::endl;
    solveRigidMotion(X2e, X1e, T1);
    std::cout << "T1.matrix()=" << T1.matrix() << std::endl;
    std::cout << "frobenious norm:" << (T.matrix()-T1.matrix()).squaredNorm() << std::endl;
    BOOST_CHECK_SMALL((T.matrix()-T1.matrix()).squaredNorm(), (float)1e-12);
    MatrixXf X2_est = T1.matrix()*X1;
    BOOST_CHECK_SMALL((X2-X2_est).squaredNorm(), (float)1e-12);
}
*/
#if 0
bool test_P_from_KRt()
{
    
    LoggerPtr logger(Logger::getLogger("mvg.test_P_from_KRt"));

    /* Camera intrinsics matrix */
    Mat K = (Mat_<float>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    
    /*  Rotation from camera coords into the world coords */
    Mat R = (Mat_<float>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    
    /* camera center in world coords */
    Mat C_Cam = (Mat_<float>(3,1) << 0, 0, 0);

    /* 3d point in (homogenious) world coordinates */
    Mat X = (Mat_<float>(3,1) << 3, 3, 3), X_h(4, X.cols, X.type());

    Mat P = P_from_KRt<float>(K,R,Mat(-R*C_Cam));
    LOG4CXX_DEBUG(logger, "camera matrix: " + _str<float>(P));
    Mat p1 = P*e2h<float>(X);
    LOG4CXX_DEBUG(logger, "projected point(s): " + _str<float>(p1));
    /* coords of X_Cam in camera coords */
    Mat X_Cam = R*(X-C_Cam);
    Mat p2 = K*X_Cam;
    LOG4CXX_DEBUG(logger, "projected points (direct computation):" + _str<float>(p2));
    Mat rms_err = rms<float>(p1, p2);
    LOG4CXX_DEBUG(logger, "rms: " + _str<float>(rms_err));
    return rms_err.at<float>(0,0) < 1e-3;
}
#endif
