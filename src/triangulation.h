#ifndef _TRIANGULATION_H
#define _TRIANGULATION_H

#include <vector>
#include <opencv2/core/core.hpp>

#include "logger.h"

using namespace std;
using namespace cv;

vector<Point3f>
triangulate_dlt(const vector<Point2f>& x1,
		const vector<Point2f>& x2,
		Mat P1, Mat P2);
#endif
