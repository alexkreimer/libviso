#include "misc.h"

bool isEqual(double x, double y)
{
    const double epsilon = 1e-6;
    return std::abs(x - y) <= epsilon * std::abs(x);
  // see Knuth section 4.2.2 pages 217-218
}

bool isEqual(float x, float y)
{
    const float epsilon = 1e-6;
    return std::abs(x - y) <= epsilon * std::abs(x);
  // see Knuth section 4.2.2 pages 217-218
}
