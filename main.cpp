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
#include <vector>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

static tbb::spin_rw_mutex blur_m, w_m;

Mat create_circular_mask( int c , double r = 0) {
    int center = int( double(c) / 2 );

    int radius;
    if ( r == 0 ) {
        radius = std::min( center, c - center );
    }
    else {
        radius = r;
    }


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
//    boxFilter(depth_map, depth_map, -1, Size(5,5));

    cv::Scalar mean, stddev;
    cv::meanStdDev(depth_map, mean, stddev);
    double depth_std = stddev[0];
    Mat kernel = imread(kernel_path,IMREAD_GRAYSCALE);
    kernel.convertTo(kernel, CV_64F, 1./255.);


    Mat img = imread( input_path, CV_64F );
    img.convertTo(img, CV_64F);

    cv::pow(img, 3., img);

    Mat blur_layer = Mat(cv::Size(img.size[1], img.size[0]), CV_64FC3, Scalar(0. ,0., 0.));
    Mat w = Mat(cv::Size(img.size[1], img.size[0]), CV_64FC3, Scalar(1. ,1., 1.));

    double DoF = compute_dof(f_depth, weight_dof);

    tbb::parallel_for(0, img.rows, [&](int i){
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
            resize(kernel,k,Size(p_coc + 1, p_coc + 1),0,0,INTER_NEAREST);

            Mat add_mask_depth = depth_map( Range(minx, maxx+1), Range(miny, maxy+1)  )  > p_depth - depth_std;

            add_mask_depth.convertTo(add_mask_depth, CV_64F, 1./255.);

            Mat mask;

            cv::multiply( add_mask_depth, k( Range( cmaxx-(maxx+1-minx), cmaxx ), Range(cmaxy-(maxy+1-miny), cmaxy) ), mask );

            Mat mask_3d;
            Mat in[] = {mask, mask, mask};
            merge(in, 3, mask_3d);
            mask_3d.convertTo(mask_3d, CV_64FC3);

            blur_m.lock();
            w( Range(minx, maxx+1), Range(miny, maxy+1) ) += mask_3d;
            blur_m.unlock();

            Mat p_color_map = Mat( Size( (maxy+1-miny), (maxx+1-minx) ), CV_64FC3, Scalar(img.at<Vec3d>(i,j)) );
            w_m.lock();
            blur_layer( Range( minx, maxx+1), Range(miny, maxy+1) ) += mask_3d.mul(p_color_map) ;
            w_m.unlock();

        });
    });


    Mat result = (img + blur_layer) / w;
    cv::pow(result, 1./3., result);

    result.convertTo(result, CV_8U);

    return result;
}


Mat bokeh_rendering_approx( string input_path, string depth_map_path, double f_depth, double weight_coc = 0.05, double weight_dof = 1, string kernel_path = "", int level = 12, double bokeh_level = 3 ) {

    Mat depth_map = imread( depth_map_path, IMREAD_GRAYSCALE );
    depth_map.convertTo(depth_map, CV_64F);
    depth_map = 1 - (depth_map / 255.);


    Mat kernel = imread(kernel_path,IMREAD_GRAYSCALE);
    cv::flip(kernel, kernel, -1);

    Mat img = imread( input_path, CV_64F );
    img.convertTo(img, CV_64F);

    cv::pow(img, bokeh_level, img);

    std::vector<Mat> src(level), result(level), level_mask(level);
    Mat aggregate_mask;

    double DoF = compute_dof(f_depth, weight_dof);

    Mat w = Mat(cv::Size(img.size[1], img.size[0]), CV_64FC3, Scalar(1. ,1., 1.));

    tbb::parallel_for(0, level, [&](int i){
        double lbound, hbound;
        lbound = 1./double(level) * i;
        hbound = 1./double(level) * (i + 1);

        Mat mask;
        if ( i != level - 1) {
            cv::inRange( depth_map, lbound, hbound-0.00001, mask );
        }
        else {
            cv::inRange( depth_map, lbound, hbound, mask );
        };

        mask.convertTo(mask, CV_8U);
        mask.copyTo(level_mask.at(i));
    });

    tbb::parallel_for(0, level, [&](int i){
        double lbound, hbound;
        lbound = 1./double(level) * i;
        hbound = 1./double(level) * (i + 1);

        Mat mask = level_mask.at(i);

        img.copyTo( src.at(i));
        src.at(i).convertTo(src.at(i), CV_64F);

        double d_p_coc = compute_coc( (lbound + hbound)/2, f_depth, DoF, weight_coc );
        int p_coc = int(d_p_coc * ( img.size[0] + img.size[1] ) * 0.5);
        if ( p_coc == 0 || p_coc == 1 ) {
            src.at(i).copyTo(result.at(i), mask);
            return ;
        }
        if ( p_coc % 2 ==0 ) {
            p_coc += 1;
        }


        Mat k;

        resize(kernel,k,Size(p_coc, p_coc),0,0,INTER_LINEAR);
        k.convertTo(k, CV_64F);

        k = k / cv::sum(k  )[0] ;

        Mat inter_result;

        cv::filter2D(src.at(i), inter_result, -1, k);

        Mat masked_img_1 = img.clone();

        Mat masked_img;
        masked_img_1.convertTo(masked_img_1,CV_64FC3, 1, 10);
        masked_img_1.copyTo(masked_img, mask);


        Mat masked_blur_img;
        cv::filter2D(masked_img, masked_blur_img, -1, k);

        Mat blur_edge_mask;
        cv::inRange(masked_blur_img, Scalar(0.001, 0.001, 0.001), Scalar(DBL_MAX, DBL_MAX, DBL_MAX), blur_edge_mask );

        tbb::spin_rw_mutex blur_edge_m;

        tbb::parallel_for(0, i, [&](int j) {

            double j_lbound, j_hbound, j_depth;
            j_lbound = 1./double(level) * j;
            j_hbound = 1./double(level) * (j + 1);
            j_depth = (j_lbound + j_hbound)/2;

            if (  j_depth + 2./level < (lbound + hbound)/2 ) {
                Mat not_mask_j;
                cv::bitwise_not(level_mask.at(j), not_mask_j);

                blur_edge_m.lock();

                cv::bitwise_and(  not_mask_j, blur_edge_mask, blur_edge_mask );

                blur_edge_m.unlock();
            }


        });

        inter_result.copyTo(result.at(i), blur_edge_mask);

        Mat mask_3d;
        Mat in[] = {mask, mask, mask};
        merge(in, 3, mask_3d);
        mask_3d.convertTo(mask_3d, CV_8UC3);

        Mat blur_edge_mask_3d;
        Mat in2[] = {blur_edge_mask, blur_edge_mask, blur_edge_mask};
        merge(in2, 3, blur_edge_mask_3d);
        blur_edge_mask_3d.convertTo(blur_edge_mask_3d, CV_8UC3);

        Mat outer_mask;


        Mat not_mask_3d;
        cv::bitwise_not(mask_3d, not_mask_3d);

        cv::bitwise_and(  blur_edge_mask_3d, not_mask_3d, outer_mask );

        outer_mask.convertTo(outer_mask, CV_64FC3, 1./255.);


        w_m.lock();
        w += outer_mask;
        w_m.unlock();


    });

    Mat final_result = result.at(0).clone();

    final_result.convertTo(final_result, CV_64F);

    for( int i = 1; i < level; i ++  ) {

        final_result += result.at(i);
    }

    final_result = final_result / w;

    cv::pow(final_result, 1./bokeh_level, final_result);

    final_result.convertTo(final_result, CV_8U);

    return final_result;

}


int main() {

    clock_t start,end;
    start=clock();
    Mat result = bokeh_rendering_approx("H:\\ECE496\\blur\\Bokeh_from_depth\\people.jpg", "H:\\ECE496\\blur\\Bokeh_from_depth\\depth_people.png", 0.9, 0.02 ,1, "H:\\ECE496\\blur\\Bokeh_from_depth\\kernel_3.png", 24);
    end=clock();
    cout<<"Time spent: "<<(double)(end-start)/CLOCKS_PER_SEC<<endl;

    imwrite("H:\\ECE496\\blur\\Bokeh_from_depth\\result.png", result);



    return 0;
}
