#include <boost/assert.hpp>
#include <boost/format.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "estimation.h"

using Eigen::MatrixXf;
using Eigen::Matrix3f;
using Eigen::Vector3f;
using Eigen::JacobiSVD;
using Eigen::Affine3f;
using std::cout;
using std::endl;

/*
 * Orthogonal Procrustes problem:
 *    R,t = min_{R,t} \sum_i(Rx_i+t-y_i)^2 s.t. R'R=I, det(R)=1, t \in R^3
 *
 * May be rewritten as: min_R \norm_F(A_zm*R-B_zm) 
 * s.t. A = [x1,x2,...]; A_zm = A-A.rowwise().mean();
 *      B = [y1,y2,...]; B_zm = B-B.rowwise().mean();
 * Solution: 
 * Let USV' = svd(A_zm*B_zm');
 * R = U*diag(1,1,det(UV'))*V'; 
 * t = B.rowwise().mean() - R*A.rowwise.mean();
 */
void
solveRigidMotion(const MatrixXf& A, const MatrixXf& B, Affine3f& T)
{
    BOOST_ASSERT_MSG(A.cols() > 1,
                     (boost::format("A.cols()=%d") % A.cols()).str().c_str());

    BOOST_ASSERT_MSG(A.cols() == B.cols(),
                     (boost::format("A.cols()=%d, B.cols()=%d") % A.cols() % B.cols()).str().c_str());
    BOOST_ASSERT_MSG(A.rows() == B.rows(),
                     (boost::format("A.rows()=%d, B.rows()=%d") % A.rows() % B.rows()).str().c_str());
    BOOST_ASSERT_MSG(A.rows() == 3, (boost::format("A.rows()=%d") % A.rows()).str().c_str());
    
    Vector3f mean1 = A.rowwise().mean(), mean2 = B.rowwise().mean();
    MatrixXf A_zm = A.colwise()-mean1, B_zm = B.colwise()-mean2, C = A_zm*B_zm.transpose();
    JacobiSVD<MatrixXf> svd = C.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    Matrix3f UVt = svd.matrixU()*svd.matrixV().transpose();
    Vector3f v;
    v << 1, 1, UVt.determinant();
    Matrix3f R = svd.matrixU()*v.asDiagonal()*svd.matrixV().transpose();
    Vector3f t = mean1 - R*mean2;
    T = R;
    T.translation() = t;
}
