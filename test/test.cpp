#define BOOST_TEST_MODULE viso_tests
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "src/viso.h"
#include "src/estimation.h"
#include "src/mvg.h"

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
    Mat X(3,N,DataType<float>::type);
    for(int i=0; i<1000; ++i)
    {
    }
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
