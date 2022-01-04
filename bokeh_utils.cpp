//
// Created by zxc21 on 2021/11/16.
//

#include "bokeh_utils.h"
#include <torch/torch.h>
#include <torch/script.h>

struct level_info {
    double lbound;
    double hbound;
    int coc;
    level_info(double Lbound, double Hbound, int Coc) {
        lbound = Lbound; hbound = Hbound; coc = Coc;
    }
};



cv::Mat create_circular_mask( int c , double r) {

    int center = int( double(c) / 2 );

    int radius;
    if ( r == 0 ) {
        radius = std::min( center, c - center );
    }
    else {
        radius = r;
    }


    cv::Mat X ( cv::Size(c,1),CV_64F );

    for ( int i = 0; i < c; i ++ )
        X.at<double>(0, i) = double(i);

    cv::Mat Y = X.t();

    X = (X - center).mul(X - center);
    Y = (Y - center).mul(Y - center);

    cv::Mat Xb, Yb;

    repeat(X, c, 1, Xb);
    repeat(Y, 1, c, Yb);

    cv::Mat dist_from_center;
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
uchar* bokeh_rendering_approx_h( int rows, int cols, uchar* img_src_p, uchar* depth_map_src_p, double f_depth, double weight_coc, double weight_dof, const char * kernel_path, int level, double bokeh_level, bool lap_bl ) {

    cv::Mat depth_map = cv::Mat(rows,cols,CV_64F,depth_map_src_p);
    depth_map = 1 - (depth_map / 255.);

    cv::Mat kernel = cv::imread(kernel_path,cv::IMREAD_GRAYSCALE);
    cv::flip(kernel, kernel, -1);

    cv::Mat img = cv::Mat(rows,cols,CV_64FC3,img_src_p);
    cv::pow(img, bokeh_level, img);


    std::vector<level_info> level_infos;
    cv::Mat aggregate_mask;

    double DoF = compute_dof(f_depth, weight_dof);


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


    std::vector<cv::Mat> src(level_count), result(level_count), level_mask(level_count), level_notmask(level_count), blur_mask(level_count);

// Calculate mask for each depth
    tbb::parallel_for(0, level_count, [&](int i){
        double lbound = level_infos.at(i).lbound;
        double hbound = level_infos.at(i).hbound;

        cv::Mat mask;
        if ( i != level_count - 1) {
            cv::inRange( depth_map, lbound, hbound-0.00001, mask );
        }
        else {
            cv::inRange( depth_map, lbound, hbound, mask );
        };

        mask.convertTo(mask, CV_8U);
        mask.copyTo(level_mask.at(i));
    });



    tbb::parallel_for(0, level_count, [&](int i){

        int p_coc = level_infos.at(i).coc;
        cv::Mat mask = level_mask.at(i);

        img.copyTo( src.at(i));
        src.at(i).convertTo(src.at(i), CV_64F);

        cv::Mat k;
        resize(kernel,k,cv::Size(p_coc, p_coc),0,0,cv::INTER_LINEAR);
        k.convertTo(k, CV_64F);
        k = k / cv::sum(k  )[0] ;

        cv::filter2D(src.at(i), result.at(i), -1, k);
        cv::pow(result.at(i), 1./bokeh_level, result.at(i));
        result.at(i).convertTo(result.at(i),CV_16S);

        cv::Mat masked_img;

        mask.convertTo( masked_img, CV_64F );


        cv::Mat masked_blur_img;

        cv::filter2D(masked_img, masked_blur_img, -1, k);


        cv::inRange(masked_blur_img, cv::Scalar(0.0001), cv::Scalar(DBL_MAX), blur_mask.at(i) );


    });





    cv::Mat final_result = result.at(0);
    cv::Mat c_mask = blur_mask.at(0);


    cv::detail::Blender * blender;
    if ( lap_bl )
        blender = new cv::detail::MultiBandBlender(false, 2 );
    else
        blender = new cv::detail::Blender;

    blender->prepare(cv::Rect(0, 0, final_result.size().width, final_result.size().height));


    blender->feed(final_result, c_mask, cv::Point2f (0,0));



    for ( int i = 1; i < level_count; i ++ ) {

        cv::Mat in_mask; cv::bitwise_not(c_mask, in_mask);


        cv::bitwise_and( in_mask, blur_mask.at(i), in_mask );

//        Mat t_image; result.at(i).copyTo(t_image, in_mask);
//        final_result += t_image;

        blender->feed(result.at(i), in_mask, cv::Point2f (0,0));


        cv::bitwise_or( c_mask, blur_mask.at(i), c_mask );


    }

    blender->blend(final_result, cv::Mat(cv::Size(final_result.size[1], final_result.size[0]), CV_8U, cv::Scalar(255)));



    final_result.convertTo(final_result, CV_8U);


    uchar* buffer = (uchar*)malloc(sizeof(uchar)*final_result.rows*final_result.cols*3);
    memcpy(buffer, final_result.data, final_result.rows*final_result.cols*3);

    return buffer;
}

uchar* bokeh_rendering_approx_h_gpu( int rows, int cols, uchar* img_src_p, uchar* depth_map_src_p, double f_depth, double weight_coc, double weight_dof, const char * kernel_path, int level, double bokeh_level, bool lap_bl ) {

    cv::Mat depth_map = cv::Mat(rows,cols,CV_64F,depth_map_src_p);
    depth_map = 1 - (depth_map / 255.);

    cv::Mat kernel = cv::imread(kernel_path);
    cv::flip(kernel, kernel, -1);


    cv::Mat img = cv::Mat(rows,cols,CV_64FC3,img_src_p);
    img.convertTo(img, CV_8UC3);
    torch::Tensor img_tensor = mat2Tensor3d( img ).cuda();


    std::vector<level_info> level_infos;

    double DoF = compute_dof(f_depth, weight_dof);
    clock_t start,end;


    int level_count = 0;
    int max_p_coc = -1;
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

        if ( p_coc > max_p_coc ) {
            max_p_coc = p_coc;
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


    torch::nn::Conv2d filters(torch::nn::Conv2dOptions(4 * level_count, 4 * level_count, max_p_coc).stride(1).bias(false).groups(4 * level_count).padding((max_p_coc - 1)/2).padding_mode(torch::kReflect));
    filters->to(torch::kCUDA);
    torch::Tensor images_and_masks = torch::zeros( { 1, 4 * level_count, img.rows, img.cols }).cuda();



// Calculate mask for each depth
    tbb::parallel_for(0, level_count, [&](int i){
        double lbound = level_infos.at(i).lbound;
        double hbound = level_infos.at(i).hbound;

        cv::Mat mask;
        if ( i != level_count - 1) {
            cv::inRange( depth_map, lbound, hbound-0.00001, mask );
        }
        else {
            cv::inRange( depth_map, lbound, hbound, mask );
        };

        mask.convertTo(mask, CV_8U);

        torch::Tensor mask_tensor = mat2Tensor2d( mask ).cuda();
        images_and_masks.index_put_( {torch::indexing::Slice( 0,1 , 1  ), torch::indexing::Slice( 4 * i,4 * i + 3 , 1  ), "..."},  img_tensor);
        images_and_masks.index_put_( {torch::indexing::Slice( 0,1 , 1  ), torch::indexing::Slice( 4 * i + 3,4 * i + 4 , 1  ), "..."},  mask_tensor);
    });

    tbb::parallel_for(0, level_count, [&](int i){

        int p_coc = level_infos.at(i).coc;

        cv::Mat k;
        resize(kernel,k,cv::Size(p_coc, p_coc) );

        torch::Tensor kernel_tensor = mat2Tensor3d( k ).cuda();
        int pad_size = (max_p_coc - p_coc) / 2;
        kernel_tensor = torch::nn::functional::pad(kernel_tensor, torch::nn::functional::PadFuncOptions( { pad_size,pad_size, pad_size,pad_size, 0, 0, 0, 0 } ).mode(torch::kConstant).value(0) );
        kernel_tensor = kernel_tensor.permute( { 1, 0, 2, 3 }   );

        torch::Tensor kernel_tensor_gray = (kernel_tensor.index({0, "..."}) + kernel_tensor.index({1, "..."}) + kernel_tensor.index({2, "..."})) / 3. ;

        filters->weight.data().index_put_( { 4 * i, "..." }, kernel_tensor.index({0, "..."}) / kernel_tensor.index({0, "..."}).sum() );
        filters->weight.data().index_put_( { 4 * i + 1, "..." }, kernel_tensor.index({1, "..."}) / kernel_tensor.index({1, "..."}).sum() );
        filters->weight.data().index_put_( { 4 * i + 2, "..." }, kernel_tensor.index({2, "..."}) / kernel_tensor.index({2, "..."}).sum() );
        filters->weight.data().index_put_( { 4 * i + 3, "..." }, kernel_tensor_gray / kernel_tensor_gray.sum() );

    });


    torch::Tensor results_tensor = filters( images_and_masks.pow(3) ).pow( 1./3. );
    results_tensor = results_tensor.detach().to(torch::kCPU).squeeze(0);

    torch::Tensor f_result = results_tensor.index( { torch::indexing::Slice( 0, 3, 1 ), "..." } ).unsqueeze(0);
    cv::Mat final_result = tensor2Mat3d( f_result );
    torch::Tensor f_mask = results_tensor.index( { torch::indexing::Slice( 3, 4, 1 ), "..." } ).unsqueeze(0);
    cv::Mat c_mask = tensor2Mat2d( f_mask );
    c_mask.convertTo(c_mask, CV_64F);
    cv::inRange(c_mask, cv::Scalar(0.0001), cv::Scalar(DBL_MAX), c_mask );


    cv::detail::Blender * blender;
    if ( lap_bl )
        blender = new cv::detail::MultiBandBlender(false, 2 );
    else
        blender = new cv::detail::Blender;

    blender->prepare(cv::Rect(0, 0, final_result.size().width, final_result.size().height));
    blender->feed(final_result, c_mask, cv::Point2f (0,0));


    for ( int i = 1; i < level_count; i ++ ) {

        cv::Mat in_mask; cv::bitwise_not(c_mask, in_mask);

        torch::Tensor i_result_tensor = results_tensor.index( { torch::indexing::Slice( 4 * i, 4 * i + 3, 1 ), "..." } ).unsqueeze(0);
        cv::Mat i_result = tensor2Mat3d( i_result_tensor );

        torch::Tensor i_mask_tensor = results_tensor.index( { torch::indexing::Slice( 4 * i + 3, 4 * i + 4, 1 ), "..." } ).unsqueeze(0);
        cv::Mat i_mask = tensor2Mat2d( i_mask_tensor );
        i_mask.convertTo(i_mask, CV_64F);
        cv::inRange(i_mask, cv::Scalar(0.0001), cv::Scalar(DBL_MAX), i_mask);


        cv::bitwise_and( in_mask, i_mask, in_mask );


        blender->feed(i_result, in_mask, cv::Point2f (0,0));


        cv::bitwise_or( c_mask, i_mask, c_mask );


    }



    blender->blend(final_result, cv::Mat(cv::Size(final_result.size[1], final_result.size[0]), CV_8U, cv::Scalar(255)));
    final_result.convertTo(final_result, CV_8U);


    uchar* buffer = (uchar*)malloc(sizeof(uchar)*final_result.rows*final_result.cols*3);
    memcpy(buffer, final_result.data, final_result.rows*final_result.cols*3);

    return buffer;
}

cv::Mat bokeh_rendering( string input_path, string depth_map_path, double f_depth, double weight_coc, double weight_dof, const string& kernel_path ) {

    tbb::spin_rw_mutex blur_m, w_m;

    cv::Mat depth_map = imread( depth_map_path, cv::IMREAD_GRAYSCALE );
    depth_map.convertTo(depth_map, CV_64F);
    depth_map = 1 - (depth_map / 255.);

    cv::Scalar mean, stddev;
    cv::meanStdDev(depth_map, mean, stddev);
    double depth_std = stddev[0];
    cv::Mat kernel = imread(kernel_path,cv::IMREAD_GRAYSCALE);
    kernel.convertTo(kernel, CV_64F, 1./255.);


    cv::Mat img = cv::imread( input_path, CV_64F );
    img.convertTo(img, CV_64F);

    cv::pow(img, 3., img);

    cv::Mat blur_layer = cv::Mat(cv::Size(img.size[1], img.size[0]), CV_64FC3, cv::Scalar(0. ,0., 0.));
    cv::Mat w = cv::Mat(cv::Size(img.size[1], img.size[0]), CV_64FC3, cv::Scalar(1. ,1., 1.));

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

            cv::Mat k;
            resize(kernel,k,cv::Size(p_coc + 1, p_coc + 1),0,0,cv::INTER_NEAREST);

            cv::Mat add_mask_depth = depth_map( cv::Range(minx, maxx+1), cv::Range(miny, maxy+1)  )  > p_depth - depth_std;

            add_mask_depth.convertTo(add_mask_depth, CV_64F, 1./255.);

            cv::Mat mask;

            cv::multiply( add_mask_depth, k( cv::Range( cmaxx-(maxx+1-minx), cmaxx ), cv::Range(cmaxy-(maxy+1-miny), cmaxy) ), mask );

            cv::Mat mask_3d;
            cv::Mat in[] = {mask, mask, mask};
            merge(in, 3, mask_3d);
            mask_3d.convertTo(mask_3d, CV_64FC3);

            blur_m.lock();
            w( cv::Range(minx, maxx+1), cv::Range(miny, maxy+1) ) += mask_3d;
            blur_m.unlock();

            cv::Mat p_color_map = cv::Mat( cv::Size( (maxy+1-miny), (maxx+1-minx) ), CV_64FC3, cv::Scalar(img.at<cv::Vec3d>(i,j)) );
            w_m.lock();
            blur_layer( cv::Range( minx, maxx+1), cv::Range(miny, maxy+1) ) += mask_3d.mul(p_color_map) ;
            w_m.unlock();

        });
    });


    cv::Mat result = (img + blur_layer) / w;
    cv::pow(result, 1./3., result);

    result.convertTo(result, CV_8U);

    return result;
}

void release(uchar* data) {
    free(data);
}


torch::Tensor mat2Tensor3d(cv::Mat const& src)
{
    cv::Mat p_src; src.convertTo(p_src, CV_8UC3);
    int w = p_src.rows, h = p_src.cols;
    std::vector<int64_t> sizes = {w * h * 3};
    torch::Tensor input_tensor = torch::zeros( sizes ).toType(torch::kU8);
    std::memcpy(  (void *) input_tensor.data_ptr(), (void *) src.data, sizeof(torch::kU8) * input_tensor.numel() );
    return input_tensor.toType(torch::kF32).reshape({w, h ,3}).permute({2, 0, 1}).unsqueeze(0);
}


cv::Mat tensor2Mat3d(torch::Tensor &src) {
    torch::Tensor p_src = src.squeeze().detach().permute({1, 2, 0});
    p_src = p_src.to(torch::kU8);
    p_src = p_src.to(torch::kCPU);
    cv::Mat resultImg(p_src.size(0), p_src.size(1), CV_8UC3);
    p_src = p_src.reshape( {p_src.size(0)*p_src.size(1)*3}  );
    std::memcpy((void *) resultImg.data, p_src.data_ptr(), sizeof(torch::kU8) * p_src.numel());
    return resultImg;
}


torch::Tensor mat2Tensor2d(cv::Mat const& src)
{
    cv::Mat p_src; src.convertTo(p_src, CV_8U);
    int w = p_src.rows, h = p_src.cols;
    std::vector<int64_t> sizes = {w * h};
    torch::Tensor input_tensor = torch::zeros( sizes ).toType(torch::kU8);
    std::memcpy(  (void *) input_tensor.data_ptr(), (void *) src.data, sizeof(torch::kU8) * input_tensor.numel() );
    return input_tensor.toType(torch::kF32).reshape({w, h}).unsqueeze(0).unsqueeze(0);
}


cv::Mat tensor2Mat2d(torch::Tensor &src) {
    torch::Tensor p_src = src.squeeze().squeeze().detach();
    p_src = p_src.to(torch::kU8);
    p_src = p_src.to(torch::kCPU);
    cv::Mat resultImg(p_src.size(0), p_src.size(1), CV_8U);
    p_src = p_src.reshape( {p_src.size(0)*p_src.size(1)}  );
    std::memcpy((void *) resultImg.data, p_src.data_ptr(), sizeof(torch::kU8) * p_src.numel());
    return resultImg;
}

void cudaWarmup() {
    torch::Tensor a = torch::rand( {100, 100}, torch::Device(torch::kCUDA) ), b = torch::rand( {100, 100}, torch::Device(torch::kCUDA) );
    torch::Tensor re = torch::matmul(a, b);
    cout << "Warmup finished, key = " << re.max().item<float>() << endl;
}
