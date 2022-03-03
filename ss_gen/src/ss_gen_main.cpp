#define _CRT_SECURE_NO_WARNINGS

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
#include <windows.h>

#else
#include <dlfcn.h>
typedef void* HINSTANCE;

#endif

// C/C++ includes
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <list>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/video.hpp>
//#include <opencv2/imgcodecs.hpp>

// custom includes
#include <cv_random_image_gen.h>
//#include <cv_create_gaussian_kernel.h>
//#include <cv_dft_conv.h>
#include <read_params.h>
#include <num2string.h>
#include <file_ops.h>

//#include <vs_gen_lib.h>

// ----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& out, std::vector<uint8_t>& item)
{
    for (uint64_t idx = 0; idx < item.size() - 1; ++idx)
    {
        out << static_cast<uint32_t>(item[idx]) << ",";
    }
    out << static_cast<uint32_t>(item[item.size() - 1]);
    return out;
}

// ----------------------------------------------------------------------------
template <typename T>
inline std::ostream& operator<<(std::ostream& out, std::vector<T>& item)
{
    for (uint64_t idx = 0; idx < item.size() - 1; ++idx)
    {
        out << item[idx] << ",";
    }
    out << item[item.size() - 1];
    return out;
}

// ----------------------------------------------------------------------------
inline int32_t calc_shift(double range, double focal_length, double pixel_size, double baseline)
{

}

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    uint32_t idx = 0, jdx = 0;
    uint32_t img_h = 512;
    uint32_t img_w = 512;
    //cv::Size img_size(img_h, img_w);

    cv::RNG rng(time(NULL));

    // timing variables
    typedef std::chrono::duration<double> d_sec;
    auto start_time = std::chrono::system_clock::now();
    auto stop_time = std::chrono::system_clock::now();
    auto elapsed_time = std::chrono::duration_cast<d_sec>(stop_time - start_time);
    std::string platform;

    cv::Mat random_img;
    cv::Mat img_left, img_right;
    cv::Mat layer_left, layer_right;
    cv::Mat mask_left, mask_right;
    cv::Mat output_img, output_mask;
    cv::Mat montage;
    cv::Mat depth_map;

    std::string montage_window = "montage";

    std::string input_param_file;

    std::string scenario_name;
    std::string f1_filename, f2_filename, depthmap_filename;

    int32_t N, min_N, max_N;
    uint8_t bg_dm, fg_dm;                   // background and foreground depthmap values
    double bg_prob, fg_prob;                // background and foreground probability of this value being selected
    double bg_x, fg_x;
    std::vector<double> bg_range_values, fg_range_values;

    std::vector<uint8_t> dm_values;         // container to store the current set of depthmap values
    std::vector<uint8_t> dm_indexes;
    double shape_scale = 0.1;
    double pattern_scale = 0.1;
    double tmp_shape_scale;
    uint32_t bg_shape_num;                  // number of shapes in teh background image
    int32_t x_offset, y_offset;             // used to jitter the right hand side image slightly
    
    // stereo camera parameters (meters)
    double pixel_size = 2e-9;
    double focal_length = 0.00212;
    double baseline = 0.120;
    std::vector<double> ranges;
    std::vector<uint32_t> disparity;
    std::vector<uint32_t> bg_disparity, fg_disparity;

    uint8_t dataset_type = 0;
    uint32_t max_dm_vals_per_image = 8;
    uint32_t num_images = 10;
    std::string save_location = "../results/test/";
    //std::string lib_filename;

    if (argc == 1)
    {
        std::cout << "Error: Missing confige file" << std::endl;
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        std::cout << "Usage: ss_gen <confige_file.txt>" << std::endl;
#else
        std::cout << "Usage: ./ss_gen <confige_file.txt>" << std::endl;
#endif
        std::cout << std::endl;
        std::cin.ignore();
        return 0;
    }

    uint8_t HPC = 0;
    get_platform(platform);
    if (platform.compare(0, 3, "HPC") == 0)
    {
        std::cout << "HPC Platform Detected." << std::endl;
        HPC = 1;
    }

    std::string param_filename = argv[1];
    read_params(param_filename, scenario_name, bg_prob, bg_range_values, fg_prob, fg_range_values, ranges, pixel_size, focal_length, 
        baseline, img_h, img_w, max_dm_vals_per_image, num_images, save_location);

    bg_shape_num = (uint32_t)std::floor(1.9 * std::max(img_w, img_h));

    // ----------------------------------------------------------------------------
    // check to make sure that min value is not 0, if it is erase it for each of the ranges
    if (ranges[0] == 0.0)
    {
        ranges.erase(ranges.begin());
    }

    if (bg_range_values[0] == 0.0)
    {
        bg_range_values.erase(bg_range_values.begin());
    }

    if (fg_range_values[0] == 0.0)
    {
        fg_range_values.erase(fg_range_values.begin());
    }

    // ----------------------------------------------------------------------------
    // assign the foreground and background depthmap values based on the ranges input
    uint8_t fg_dm_value = 0;
    uint8_t bg_dm_value = static_cast<uint8_t>(ranges.size());

    // ----------------------------------------------------------------------------
    // calculate the disparity values based on the binned ranges, focal length, baseline and pixel size for ROI, background and foreground
    for (idx = 0; idx < ranges.size(); ++idx)
    {
        disparity.push_back((uint32_t)(floor((focal_length * baseline) / (ranges[idx] * pixel_size) + 0.5)));
    }

    for (idx = 0; idx < bg_range_values.size(); ++idx)
    {
        bg_disparity.push_back((uint32_t)(floor((focal_length * baseline) / (bg_range_values[idx] * pixel_size) + 0.5)));
    }

    for (idx = 0; idx < fg_range_values.size(); ++idx)
    {
        fg_disparity.push_back((uint32_t)(floor((focal_length * baseline) / (fg_range_values[idx] * pixel_size) + 0.5)));
    }

    // ----------------------------------------------------------------------------
    // create results directories if they do not exist
    mkdir(save_location + "images");
    mkdir(save_location + "depth_maps");

    // ----------------------------------------------------------------------------
    // save the parameters that were used to generate the dataset
    std::ofstream param_stream(save_location + scenario_name + "parameters.txt", std::ofstream::out);
    param_stream << "# Parameters used to generate the dataset" << std::endl;
/*
    param_stream << depthmap_values << std::endl;
    param_stream << sigma_table << std::endl;
    param_stream << br1_table << std::endl;
    param_stream << br2_table << std::endl;
    param_stream << static_cast<uint32_t>(dataset_type) << std::endl;
    param_stream << img_h << "," << img_w << std::endl;
    param_stream << max_dm_num << std::endl;
    param_stream << num_objects << std::endl;
    param_stream << num_images << std::endl << std::endl;
    param_stream << "------------------------------------------------------------------" << std::endl;

    // print out the parameters
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "Parameters used to generate the dataset" << std::endl;
    std::cout << "Depthmap Values:  " << depthmap_values << std::endl;
    std::cout << "Sigma Table:      " << sigma_table << std::endl;
    std::cout << "Blur Radius 1:    " << br1_table << std::endl;
    std::cout << "Blur Radius 2:    " << br2_table << std::endl;
    std::cout << "Dataset Type:     " << static_cast<uint32_t>(dataset_type) << std::endl;
    std::cout << "Image Size (hxw): " << img_h << " x " << img_w << std::endl;
    std::cout << "DM Values/Image:  " << max_dm_num << std::endl;
    std::cout << "# of Objects:     " << num_objects << std::endl;
    std::cout << "# of Images:      " << num_images << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
*/

    // if the platform is an HPC platform then don't display anything
    if (!HPC)
    {
        // setup the windows to display the results
        cv::namedWindow(montage_window, cv::WINDOW_NORMAL);
        cv::resizeWindow(montage_window, 2 * img_w, img_h);
    }

    // do work here
    try
    {    

        std::ofstream dataLog_stream(save_location + scenario_name + "input_file.txt", std::ofstream::out);
        dataLog_stream << "# Data Directory" << std::endl;
        dataLog_stream << save_location << ", " << save_location << std::endl;
        dataLog_stream << std::endl;
        dataLog_stream << "# focus point 1 filename, focus point 2 filename, depthmap filename" << std::endl;
        
        std::cout << "Data Directory: " << save_location << std::endl;

        start_time = std::chrono::system_clock::now();

        double scale = 0.1;

        for (idx = 0; idx < num_images; ++idx)
        {
            
            //img_l = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar::all(0));
            //img_r = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar::all(0));
            //depth_map = cv::Mat(img_h, img_w, CV_8UC1, cv::Scalar::all(0));

            dm_values.clear();

            tmp_shape_scale = rng.uniform(shape_scale*0.65, shape_scale);

            // generate random dm_values that include the foreground and background values
            int32_t tmp_dm_num = max_dm_vals_per_image;

            // get the probablility that the background depthmap value will be used
            bg_x = rng.uniform(0.0, 1.0);

            // get the probability that the foreground depthmap value will be used
            fg_x = rng.uniform(0.0, 1.0);

            if (bg_x < bg_prob)
                tmp_dm_num--;

            if (fg_x < fg_prob)
                tmp_dm_num--;

            //generate_depthmap_set(min_dm_value, max_dm_value, tmp_dm_num, depthmap_values, dm_values, rng);
            generate_depthmap_index_set(ranges.size(), tmp_dm_num, dm_indexes, rng);

            // check the background probability and fill in the tables
            if (bg_x < bg_prob)
            {
                //uint16_t dm = rng.uniform(0, bg_br_table.size());
                //tmp_br1_table.push_back(bg_br_table[dm].first);
                //tmp_br2_table.push_back(bg_br_table[dm].second);

                dm_values.push_back(bg_dm_value);
            }

            // fill in the tables for the region of interest depthmap values
            for (jdx = 0; jdx < dm_indexes.size(); ++jdx)
            {
                //tmp_br1_table.push_back(br1_table[dm_indexes[jdx]]);
                //tmp_br2_table.push_back(br2_table[dm_indexes[jdx]]);
                dm_values.push_back(dm_indexes[jdx]);
            }

            // check the foreground probability and fill in the tables
            if (fg_x < fg_prob)
            {
                //uint16_t dm = rng.uniform(0, fg_br_table.size());
                //tmp_br1_table.push_back(fg_br_table[dm].first);
                //tmp_br2_table.push_back(fg_br_table[dm].second);

                dm_values.push_back(fg_dm_value);
            }

            // generate a random image
            generate_random_image(random_img, rng, img_h, img_w + disparity[dm_values[0]], bg_shape_num, pattern_scale);

            // crop the image according to the disparity
            img_left = random_img(cv::Rect(0, 0, img_w, img_h)).clone();
            img_right = random_img(cv::Rect(disparity[dm_indexes[0]], 0, img_w, img_h)).clone();
            depth_map = cv::Mat(img_h, img_w, CV_8UC1, cv::Scalar::all(dm_values[0]));

            for (jdx = 1; jdx < dm_values.size(); ++jdx)
            {
                //left_layer = img_l.clone();
                //right_layer = img_r.clone();

                x_offset = rng.uniform(-1, 1);
                y_offset = rng.uniform(-1, 1);

                // set the minimum and maximum number of objects in the scene per depth layer
                min_N = (int32_t)ceil(((bg_dm_value) / (double)(1.0 + exp(-0.365 * dm_values[idx] + (0.175 * bg_dm_value)))) + 3);
                max_N = (int32_t)ceil(2.0 * min_N);  // 2.0
                N = (uint32_t)std::ceil(rng.uniform(min_N, max_N + 1) * (std::max(img_w, img_h) / 512.0));

                // generate a random image (img_h+4, img_w+4+disparity[dm_values[jdx]])
                generate_random_image(random_img, rng, img_h + 4, img_w + 4 + disparity[dm_values[jdx]], bg_shape_num, pattern_scale);

                // generate a set of masks 
                generate_random_mask(output_mask, random_img.rows, random_img.cols, rng, N, shape_scale);
                
                // multiply random_img times mask_left
                cv::multiply(random_img, output_mask, output_img);

                mask_left = output_mask(cv::Rect(0, 2, img_w, img_h));
                mask_right = output_mask(cv::Rect(disparity[dm_indexes[jdx]] + x_offset, 2 + y_offset, img_w, img_h));

                layer_left = output_img(cv::Rect(0, 2, img_w, img_h)).clone();
                layer_right = output_img(cv::Rect(disparity[dm_indexes[jdx]] + x_offset, 2 + y_offset, img_w, img_h)).clone();

                overlay_image(img_left, layer_left, mask_left);
                overlay_image(img_right, layer_right, mask_right);

                // overlay depthmap
                overlay_depthmap(depth_map, mask_left, dm_values[jdx]);
                
            }










            //tmp_img = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar::all(0));

            //int16_t dm = rng.uniform(-1, 1);

            // if the platform is an HPC platform then don't display anything
            if (!HPC)
            {
                cv::hconcat(img_left, img_right, montage);
                cv::imshow(montage_window, montage);
                cv::waitKey(10);
            }

            f1_filename = "images/" + scenario_name + num2str<int>(jdx, "image_left_%04i.png");
            f2_filename = "images/" + scenario_name + num2str<int>(jdx, "image_right_%04i.png");
            depthmap_filename = "depth_maps/" + scenario_name + num2str<int>(jdx, "dm_%04i.png");

            //cv::imwrite(save_location + f1_filename, img_left);
            //cv::imwrite(save_location + f2_filename, img_right);
            //cv::imwrite(save_location + depthmap_filename, depth_map);

            std::cout << f1_filename << ", " << f2_filename << ", " << depthmap_filename << std::endl;
            
            // this doesn't get filled in anymore
            //std::cout << dm_values << std::endl;

            //param_stream << "image " << num2str<int>(jdx, "%03d: ") << dm_values << std::endl;
            //param_stream << "           " << tmp_br1_table << std::endl;
            //param_stream << "           " << tmp_br2_table << std::endl;

            dataLog_stream << f1_filename << ", " << f2_filename << ", " << depthmap_filename << std::endl;

        } // end of for loop

        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
        std::cout << "elapsed_time (s): " << elapsed_time.count() << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl << std::endl;

        param_stream.close();
        dataLog_stream.close();
    }
    catch(std::exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }


//    if (dataset_type == 1)
//    {
//#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
//        FreeLibrary(simplex_noise_lib);
//#else
//        dlclose(simplex_noise_lib);
//#endif
//    }


    std::cout << "End of Program.  Press Enter to close..." << std::endl;
	std::cin.ignore();
    cv::destroyAllWindows();

}   // end of main

