//
// Created by zxc21 on 2021/11/16.
//

#ifndef BOKEH_FROM_DEPTH_BOKEH_UTILS_H
#define BOKEH_FROM_DEPTH_BOKEH_UTILS_H

#endif //BOKEH_FROM_DEPTH_BOKEH_UTILS_H

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/spin_rw_mutex.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include<ctime>
#include <mutex>
#include <shared_mutex>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;

Mat create_circular_mask( int c , double r = 0);
extern "C" __declspec(dllexport) double compute_dof(  double f_depth, double weight = 1  );
extern "C" __declspec(dllexport) double compute_coc( double o_depth, double f_depth, double dof, double weight = 0.1 );
extern "C" __declspec(dllexport) uchar* bokeh_rendering_approx_h( int rows, int cols, uchar* img_src_p, uchar* depth_map_src_p, double f_depth, double weight_coc = 0.05, double weight_dof = 1, const char* kernel_path = "", int level = 12, double bokeh_level = 3, double cover = 1, int img_show = 0  );
Mat bokeh_rendering( string input_path, string depth_map_path, double f_depth, double weight_coc = 0.05, double weight_dof = 1, const string& kernel_path = "" );
extern "C" __declspec(dllexport) void release(uchar* data);

