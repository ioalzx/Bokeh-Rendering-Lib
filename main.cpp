// main.cpp

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/spin_rw_mutex.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include<ctime>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <cmath>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

static tbb::spin_rw_mutex blur_m, w_m;

Mat create_circular_mask( int c ) {
    int center = int( double(c) / 2 );

    int radius = std::min( center, c - center );

    Mat X ( Size(c,1),CV_64F );

    for ( int i = 0; i < c; i ++ )
        X.at<double>(0, i) = double(i);

    Mat Y = X.t();

    X = (X - center).mul(X - center);
    Y = (Y - center).mul(Y - center);

    Mat Xb, Yb;

    repeat(X, c, 1, Xb);
    repeat(Y, 1, c, Yb);

    Mat dist_from_center;
    cv::sqrt( Xb + Yb, dist_from_center );

    return dist_from_center <= radius;

}

double compute_dof(  double f_depth, double weight = 1  ) {
    return std::min(  double(1), weight * (  f_depth * f_depth  )  );
}

double compute_coc( double o_depth, double f_depth, double dof, double weight = 0.1 ) {
    if ( o_depth > f_depth + dof/2 ) {
        o_depth = o_depth - dof/2;
    }
    else if (  o_depth < f_depth - dof/2  ) {
        o_depth = o_depth + dof/2;
    }
    else {
        o_depth = f_depth;
    }

    double re = weight * std::abs( f_depth - o_depth  );

    return std::min( 1.0, re);
}

Mat bokeh_rendering( string input_path, string depth_map_path, double f_depth, double weight_coc = 0.05, double weight_dof = 1, string kernel_path = "" ) {
    Mat depth_map = imread( depth_map_path, IMREAD_GRAYSCALE );
    depth_map.convertTo(depth_map, CV_64F);
    depth_map = 1 - (depth_map / 255.);

    cv::Scalar mean, stddev;
    cv::meanStdDev(depth_map, mean, stddev);
    double depth_std = stddev[0];
    Mat kernel = imread(kernel_path,IMREAD_GRAYSCALE);


    Mat img = imread( input_path, CV_64F );
    img.convertTo(img, CV_64F);

    cv::pow(img, 3., img);

    Mat blur_layer = Mat(cv::Size(img.size[1], img.size[0]), CV_64FC3, Scalar(0. ,0., 0.));
    Mat w = Mat(cv::Size(img.size[1], img.size[0]), CV_64FC3, Scalar(1. ,1., 1.));

    double DoF = compute_dof(f_depth, weight_dof);




    tbb::parallel_for(0, img.rows, [&](int i){
//        cout << i << endl;
        tbb::parallel_for(0, img.cols, [&](int j){
            double p_depth = depth_map.at<double>( i, j );
            double d_p_coc = compute_coc( p_depth, f_depth, DoF, weight_coc );
            int p_coc = int(d_p_coc * ( img.size[0] + img.size[1] ) * 0.5);
            if ( p_coc == 0 ) {
                return ;
            }
            if ( p_coc % 2 !=0 ) {
                p_coc += 1;
            }

            int minx = std::max( 0, int( i - p_coc/2. ) );
            int maxx = std::min( img.size[0] - 1, int( i + p_coc/2. ) );

            int miny = std::max( 0, int( j - p_coc/2. ) );
            int maxy = std::min( img.size[1] - 1, int( j + p_coc/2. ) );

            int cmaxx = int(  std::min( p_coc + 1, p_coc + 1 - (i + p_coc/2 + 1 - img.size[0])  ));
            int cmaxy = int(  std::min( p_coc + 1, p_coc + 1 - (j + p_coc/2 + 1 - img.size[1])  ));

            Mat k;
            resize(kernel,k,Size(p_coc + 1, p_coc + 1),0,0,INTER_LINEAR);
            k.convertTo(k, CV_64F);


            Mat add_mask_depth = depth_map( Range(minx, maxx+1), Range(miny, maxy+1)  )  > p_depth - depth_std;

            add_mask_depth.convertTo(add_mask_depth, CV_64F, 1./255.);

            Mat mask;

            cv::multiply( add_mask_depth, k( Range( cmaxx-(maxx+1-minx), cmaxx ), Range(cmaxy-(maxy+1-miny), cmaxy) ), mask );

            mask.convertTo(mask, CV_8U);





            Mat w_color_map = Mat( Size( (maxy+1-miny), (maxx+1-minx) ), CV_64FC3, Scalar(1., 1., 1.) );
            Mat w_color_map_masked;
            w_color_map.copyTo( w_color_map_masked, mask);

            blur_m.lock_read();
            Mat inter_mask;
            cv::add( w_color_map_masked, w( Range(minx, maxx+1), Range(miny, maxy+1)), inter_mask );
            blur_m.unlock();

            blur_m.lock();
            inter_mask.copyTo( w( Range(minx, maxx+1), Range(miny, maxy+1) ) );
            blur_m.unlock();



            Mat p_color_map = Mat( Size( (maxy+1-miny), (maxx+1-minx) ), CV_64FC3, Scalar(img.at<Vec3d>(i,j)) );
            Mat p_color_map_masked;
            p_color_map.copyTo( p_color_map_masked, mask);

            w_m.lock_read();
            Mat inter_blur_layer;
            cv::add( blur_layer( Range( minx, maxx+1), Range(miny, maxy+1) ), p_color_map_masked, inter_blur_layer);
            w_m.unlock();

            w_m.lock();
            inter_blur_layer.copyTo( blur_layer( Range( minx, maxx+1), Range(miny, maxy+1) ) );
            w_m.unlock();



        });
    });





    Mat result = (img + blur_layer) / w;
    cv::pow(result, 1./3., result);

    result.convertTo(result, CV_8U);

    return result;
}


int main() {


    clock_t start,end;
    start=clock();
    Mat result = bokeh_rendering("H:\\ECE496\\blur\\Bokeh_from_depth\\source_small.jpg", "H:\\ECE496\\blur\\Bokeh_from_depth\\depth_small.png", 0.5, 0.06 ,1, "H:\\ECE496\\blur\\Bokeh_from_depth\\kernel_2.png");
    end=clock();
    cout<<"Time spent: "<<(double)(end-start)/CLOCKS_PER_SEC<<endl;



    imshow("Image", result);
    waitKey(0);

    imwrite("H:\\ECE496\\blur\\Bokeh_from_depth\\result.png", result);

//    Mat a = Mat::ones(Size(5,5), CV_64F);

//    a(Range(0,3), Range(0, 3)).forEach<double>([](double &p, const int * position) -> void {
//        p += 1;
//        cout<< position[0] << endl;
//    });
//
//    cout << a << endl;



//    tbb::parallel_for(0, 10, 1, [](int n) -> void {
//        cout << n << endl;
//    });





    return 0;
}
