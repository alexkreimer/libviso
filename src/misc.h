#ifndef _MISC_H_
#define _MISC_H_

#include <iostream>
#include <iomanip>
#include <cmath>  /* for std::abs(double) */
#include <opencv2/core/core.hpp>
#include <stdexcept>

using std::string;
using cv::Mat;
using std::stringstream;

bool isEqual(double x, double y);
bool isEqual(float x, float y);

template<typename T>
string _str(const Mat& m, bool include_dims=true, int truncate=16,
            int precision=2)
{
    stringstream ss;
    //ss << std::setprecision(precision) << fixed;
    if (include_dims)
    {
        ss << "(" << m.rows << "x" << m.cols << ") ";
    }
    ss << "[";
    for(int k=0, i=0; i<m.rows; ++i)
    {
        if (i>0)
            ss << " ";
        for(int j=0; j<m.cols; ++j, ++k)
        {
            ss << m.at<T>(i,j);
            if (j<m.cols-1)
                ss<<",";
            if (k==truncate)
            {
                ss << "...]";
                return ss.str();
            }
        }
        if (i<m.rows-1)
            ss << ";";
    }
    ss << "]";
    return ss.str();
}

template<class T>
Mat vcat(const Mat& m1, const Mat& m2)
{
//    assert(m1.cols() == m2.cols());
//    assert(m1.type() == m2.type());
    Mat res(m1.rows+m2.rows,m1.cols,m1.type());
    for (int i=0; i<m1.rows; ++i)
    {
        for(int j=0; j<m1.cols; ++j)
        {
            res.at<T>(i,j) = m1.at<T>(i,j);
        }
    }
    for (int k=m1.rows, i=0; i<m2.rows; ++i)
    {
        for(int j=0; j<m2.cols; ++j)
        {
            res.at<T>(k+i, j) = m2.at<T>(i, j);
        }
    }
    return res;
}

template<class T>
Mat hcat(const Mat& m1, const Mat& m2)
{
//    assert(m1.rows() == m2.rows());
//    assert(m1.type() == m2.type());
    
    Mat m(m1.rows, m1.cols+m2.cols, m1.type());
    for(int i=0; i<m1.rows; ++i)
    {
        for(int j=0; j<m1.cols+m2.cols; ++j)
        {
            m.at<T>(i, j) = (j<m1.cols) ? m1.at<T>(i, j) : m2.at<T>(i, j-m1.cols);
        }
    }
    return m;
}

template<class T>
Mat e2h(const Mat& X)
{
    Mat Xh(X.rows+1, X.cols, cv::DataType<T>::type);

    for(int i=0; i<X.rows; ++i)
    {
        for(int j=0; j<X.cols; ++j)
        {
            Xh.at<T>(i,j) = X.at<T>(i,j);
        }
    }
    for(int j=0; j<X.cols; ++j)
    {
        Xh.at<T>(Xh.rows-1,j) = 1.0;
    }
    return Xh;
}

template<class T>
Mat h2e(const Mat& X)
{
    Mat Xh(X.rows-1, X.cols, X.type());
    
    for(int i=0; i<Xh.rows; ++i)
    {
        for(int j=0; j<Xh.cols; ++j)
        {
            if (isEqual(abs(X.at<T>(X.rows-1,j)),.0f))
                throw std::overflow_error("divide by zero in h2e");
            Xh.at<T>(i,j) = X.at<T>(i,j)/X.at<T>(X.rows-1,j);
        }
    }
    return Xh;
}

#include <chrono>
template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F>
    static typename TimeT::rep execution(F const &func)
    {
        auto start = std::chrono::system_clock::now();
        func();
        auto duration = std::chrono::duration_cast< TimeT>(
            std::chrono::system_clock::now() - start);
        return duration.count();
    }
};

#endif /* _MISC_H_ */
