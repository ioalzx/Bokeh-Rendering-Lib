// main.cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include<time.h>
#include <omp.h>
#include <cmath>

using namespace std;
using namespace cv;


#pragma omp declare reduction(+ : Mat : omp_out += omp_in)\
                            initializer(omp_priv=omp_orig.clone())


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
    Mat depth_map = imread( depth_map_path, CV_64F );
    depth_map.convertTo(depth_map, CV_64F);
    depth_map = 1 - (depth_map / 255.);
    cv::Scalar mean, stddev;
    cv::meanStdDev(depth_map, mean, stddev);
    double depth_std = stddev[0];
    Mat kernel = imread(kernel_path,IMREAD_GRAYSCALE);



    Mat img = imread( input_path, CV_64F );
    img.convertTo(img, CV_64F);

    cv::pow(img, 3., img);

    Mat blur_layer = Mat::zeros(cv::Size(img.size[1], img.size[0]), CV_64FC3);
    Mat w = Mat(cv::Size(img.size[1], img.size[0]), CV_64FC3, Scalar(1. ,1., 1.));

    double DoF = compute_dof(f_depth, weight_dof);



#pragma omp parallel for reduction(+:blur_layer,w)
    for ( int i = 0;  i < img.rows; i ++) {

        for ( int j = 0; j < img.cols; j++ ){

//            img.at<Vec3d>(i, j) = img.at<Vec3d>(i, j) - Vec3d (20., 20., 20.);

            double p_depth = depth_map.at<double>( i, j );
            double d_p_coc = compute_coc( p_depth, f_depth, DoF, weight_coc );
            int p_coc = int(d_p_coc * ( img.size[0] + img.size[1] ) * 0.5);
            if ( p_coc == 0 ) {
                continue;
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
//            threshold(k, k, 127, 255, THRESH_BINARY);


//            Mat circular_mask = create_circular_mask(p_coc + 1)( Range( cmaxx-(maxx+1-minx), cmaxx ), Range(cmaxy-(maxy+1-miny), cmaxy) );

            Mat add_mask_depth = depth_map( Range(minx, maxx+1), Range(miny, maxy+1)  ) > p_depth - depth_std;

            Mat mask = (add_mask_depth/255).mul(k( Range( cmaxx-(maxx+1-minx), cmaxx ), Range(cmaxy-(maxy+1-miny), cmaxy) ) );

            mask = mask;
            mask.convertTo(mask, CV_8U);

            Mat w_color_map = Mat( Size( (maxy+1-miny), (maxx+1-minx) ), CV_64FC3, Scalar(1., 1., 1.) );
            Mat w_color_map_masked;
            w_color_map.copyTo( w_color_map_masked, mask);

            Mat inter_mask = (w_color_map_masked + w( Range(minx, maxx+1), Range(miny, maxy+1) ));
            inter_mask.copyTo( w( Range(minx, maxx+1), Range(miny, maxy+1) ) );


            Mat p_color_map = Mat( Size( (maxy+1-miny), (maxx+1-minx) ), CV_64FC3, Scalar(img.at<Vec3d>(i,j)) );
            Mat p_color_map_masked;
            p_color_map.copyTo( p_color_map_masked, mask);

            Mat inter_blur_layer = blur_layer( Range( minx, maxx+1), Range(miny, maxy+1) ) + p_color_map_masked;
            inter_blur_layer.copyTo( blur_layer( Range( minx, maxx+1), Range(miny, maxy+1) ) );



        }
    }

    Mat result = (img + blur_layer) / w;
    cv::pow(result, 1./3., result);

    result.convertTo(result, CV_8U);

    return result;
}


int main() {


    clock_t start,end;
    start=clock();
    Mat result = bokeh_rendering("H:\\ECE496\\blur\\Bokeh_from_depth\\source.jpg", "H:\\ECE496\\blur\\Bokeh_from_depth\\depth.png", 0.4, 0.03 ,0.6, "H:\\ECE496\\blur\\Bokeh_from_depth\\kernel_2.png");
    end=clock();
    cout<<"Time spent: "<<(double)(end-start)/CLOCKS_PER_SEC<<endl;



    imshow("Image", result);
    waitKey(0);

    imwrite("H:\\ECE496\\blur\\Bokeh_from_depth\\result.png", result);


    return 0;
}
