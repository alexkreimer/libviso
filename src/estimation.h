#ifndef _ESTIMATION_H_
#define _ESTIMATION_H_

#include <Eigen/Dense>
#include "viso.h"

void
solveRigidMotion(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B,
                      Eigen::Affine3f& T);

#endif /* _ESTIMATION_H_ */

