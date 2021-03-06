// main.cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include<ctime>
#include <mutex>
#include "bokeh_utils.h"
#include <torch/torch.h>
#include <torch/script.h>


using namespace std;
namespace F = torch::nn::functional;


cv::Mat bokeh_rendering_approx( string input_path, string depth_map_path, double f_depth, double weight_coc = 0.05, double weight_dof = 1, const string& kernel_path = "", int level = 12, double bokeh_level = 3, bool lap_bl = false) {

    cv::Mat depth_map = imread( depth_map_path, cv::IMREAD_GRAYSCALE );
    depth_map.convertTo(depth_map, CV_64F);

    cv::Mat img = cv::imread( input_path, CV_64F );
    img.convertTo(img, CV_64F);

    int n = kernel_path.length();
    char * kernel_path_array = (char *)malloc( sizeof(char) * (n+1) );

    strcpy( kernel_path_array, kernel_path.c_str() );

    clock_t start,end;
    start=clock();

    uchar* re = bokeh_rendering_approx_h( img.size[0], img.size[1], img.data, depth_map.data, f_depth,weight_coc, weight_dof, kernel_path_array, level, bokeh_level, lap_bl  );
    end=clock();
    cout<<"Time spent: "<<(double)(end-start)/CLOCKS_PER_SEC<<endl;

    cv::Mat reimg = cv::Mat( img.size[0], img.size[1], CV_8UC3, re );

    return reimg;


}



int main() {
    cv::Mat result = bokeh_rendering_approx(R"(H:\ECE496\blur\Bokeh_from_depth\source.jpg)", R"(H:\ECE496\blur\Bokeh_from_depth\depth.png)", 0.4, 0.03 ,1, R"(H:\ECE496\blur\Bokeh_from_depth\kernel_6.png)", 8, 3, true);
    cv::imwrite("H:\\ECE496\\blur\\Bokeh_from_depth\\result.png", result);
    free( result.data );
    return 0;
}
