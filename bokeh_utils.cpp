//
// Created by zxc21 on 2021/11/16.
//

#include "bokeh_utils.h"

struct level_info {
    double lbound;
    double hbound;
    int coc;
    level_info(double Lbound, double Hbound, int Coc) {
        lbound = Lbound; hbound = Hbound; coc = Coc;
    }
};





Mat create_circular_mask( int c , double r) {

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
double compute_dof(  double f_depth, double weight  ) {
    return std::min(  double(1), weight * (  f_depth * f_depth  )  );
}
double compute_coc( double o_depth, double f_depth, double dof, double weight ){
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
uchar* bokeh_rendering_approx_h( int rows, int cols, uchar* img_src_p, uchar* depth_map_src_p, double f_depth, double weight_coc, double weight_dof, const char * kernel_path, int level, double bokeh_level, double cover, int img_show ) {

    Mat depth_map = Mat(rows,cols,CV_64F,depth_map_src_p);
    depth_map = 1 - (depth_map / 255.);

    tbb::spin_rw_mutex w_m, r_m;

    Mat kernel = imread(kernel_path,IMREAD_GRAYSCALE);
    cv::flip(kernel, kernel, -1);

    Mat img = Mat(rows,cols,CV_64FC3,img_src_p);
    cv::pow(img, bokeh_level, img);


    std::vector<level_info> level_infos;
    Mat aggregate_mask;

    double DoF = compute_dof(f_depth, weight_dof);

    Mat w = Mat(cv::Size(img.size[1], img.size[0]), CV_64FC3, Scalar(1. ,1., 1.));


    int level_count = 0;
    for ( int i = 0; i < level; i ++ ) {
        double lbound, hbound;
        lbound = 1./double(level) * i;
        hbound = 1./double(level) * (i + 1);

        double d_p_coc = compute_coc( (lbound + hbound)/2, f_depth, DoF, weight_coc );
        int p_coc = int(d_p_coc * ( img.size[0] + img.size[1] ) * 0.5);
        if ( p_coc == 0 || p_coc == 1 ) {
            p_coc = 0;
        }
        if ( p_coc % 2 ==0 ) {
            p_coc += 1;
        }

        if ( i == 0 ) {
            level_infos.emplace_back( lbound, hbound, p_coc );
            level_count ++;
        }
        else if (  p_coc == level_infos[level_count - 1].coc  ) {
            level_infos[level_count - 1].hbound = hbound;
        }
        else {
            level_infos.emplace_back( lbound, hbound, p_coc );
            level_count ++;
        }

    }


    std::vector<Mat> src(level_count), result(level_count), level_mask(level_count), level_notmask(level_count);


    tbb::parallel_for(0, level_count, [&](int i){
        double lbound = level_infos.at(i).lbound;
        double hbound = level_infos.at(i).hbound;

        Mat mask;
        if ( i != level_count - 1) {
            cv::inRange( depth_map, lbound, hbound-0.00001, mask );
        }
        else {
            cv::inRange( depth_map, lbound, hbound, mask );
        };

        mask.convertTo(mask, CV_8U);
        mask.copyTo(level_mask.at(i));
    });


    for ( int i = 0; i < level_count; i ++ ) {

        if ( i == 0 ) {
            Mat not_mask_i;
            cv::bitwise_not(level_mask.at(i), not_mask_i);

            not_mask_i.copyTo(level_notmask.at(i));
            continue;
        }



        Mat not_mask_i;
        cv::bitwise_not(level_mask.at(i), not_mask_i);

        cv::bitwise_and(  not_mask_i, level_notmask.at(i - 1), level_notmask.at(i) );

    }


    Mat final_result = Mat(cv::Size(img.size[1], img.size[0]), CV_64FC3, Scalar(0. ,0., 0.));

    tbb::parallel_for(0, level_count, [&](int i){

        int p_coc = level_infos.at(i).coc;

        Mat mask = level_mask.at(i);

        img.copyTo( src.at(i));
        src.at(i).convertTo(src.at(i), CV_64F);


        Mat k;
        resize(kernel,k,Size(p_coc, p_coc),0,0,INTER_LINEAR);
        k.convertTo(k, CV_64F);
        k = k / cv::sum(k  )[0] ;

        Mat inter_result;
//        src.at(i).copyTo(inter_result, mask);
        cv::filter2D(src.at(i), inter_result, -1, k);


        Mat masked_img_1 = img.clone();
        Mat masked_img;
        masked_img_1.convertTo(masked_img_1,CV_64FC3, 1, 10);
        masked_img_1.copyTo(masked_img, mask);

        Mat chan[3];
        cv::split(masked_img, chan);

        masked_img = (chan[0] + chan[1] + chan[2] )/3;



        Mat masked_blur_img;
//        masked_img.copyTo(masked_blur_img, mask);
        cv::filter2D(masked_img, masked_blur_img, -1, k);



        Mat blur_edge_mask;
//        cv::inRange(masked_blur_img, Scalar(0.0001, 0.0001, 0.0001), Scalar(DBL_MAX, DBL_MAX, DBL_MAX), blur_edge_mask );
        cv::inRange(masked_blur_img, Scalar(0.0001), Scalar(DBL_MAX), blur_edge_mask );


        if ( i != 0 ) {
            cv::bitwise_and(  level_notmask.at(i - 1), blur_edge_mask, blur_edge_mask );
        }


        inter_result.copyTo(result.at(i), blur_edge_mask);

        r_m.lock();
        final_result += result.at(i);
        r_m.unlock();

        Mat outer_mask;
        cv::bitwise_and(  blur_edge_mask, level_notmask.at(i), outer_mask );

        Mat outer_mask_3d;
        Mat in[] = {outer_mask, outer_mask, outer_mask};
        merge(in, 3, outer_mask_3d);
        outer_mask_3d.convertTo(outer_mask_3d, CV_64F, 1./255.);

        w_m.lock();
        w += outer_mask_3d;
        w_m.unlock();


    });


    final_result = final_result / w;


    cv::pow(final_result, 1./bokeh_level, final_result);
    final_result.convertTo(final_result, CV_8U);


    uchar* buffer = (uchar*)malloc(sizeof(uchar)*final_result.rows*final_result.cols*3);
    memcpy(buffer, final_result.data, final_result.rows*final_result.cols*3);

    return buffer;
}

Mat bokeh_rendering( string input_path, string depth_map_path, double f_depth, double weight_coc, double weight_dof, const string& kernel_path ) {

    tbb::spin_rw_mutex blur_m, w_m;

    Mat depth_map = imread( depth_map_path, IMREAD_GRAYSCALE );
    depth_map.convertTo(depth_map, CV_64F);
    depth_map = 1 - (depth_map / 255.);

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

void release(uchar* data) {
    free(data);
}


