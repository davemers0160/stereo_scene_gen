#ifndef _READ_PARAMS_H_
#define _READ_PARAMS_H_


#include <file_parser.h>

//-----------------------------------------------------------------------------
std::string get_env_variable(std::string env_var)
{
    char* p;
    p = getenv(env_var.c_str());

    if (p == NULL)
        return "";
    else
        return std::string(p);
}

//-----------------------------------------------------------------------------
void get_platform(std::string& platform)
{
    platform = get_env_variable("PLATFORM");
}

//-----------------------------------------------------------------------------
void read_params(std::string param_filename,
    std::string &scenario_name,
    double& bg_prob,
    std::vector<double>& bg_range_values,
    double& fg_prob,
    std::vector<double>& fg_range_values,
    std::vector<double> &range_values, 
    double &pixel_size,
    double &focal_length,
    double &baseline,
    uint32_t &img_h, uint32_t &img_w,
    uint32_t &max_dm_vals_per_image,
    uint32_t &num_images,
    std::string &save_location
)
{
    uint32_t idx = 0, jdx = 0;

    std::vector<std::vector<std::string>> params;
    parse_csv_file(param_filename, params);

    for (idx = 0; idx < params.size(); ++idx)
    {
        switch (idx)
        {
            case 0:
                scenario_name = params[idx][0];
                break;

            // #1 background probability and associated ranges
            case 1:
                try {
                    bg_prob = std::stod(params[idx][0]);
                    parse_input_range(params[idx][1], bg_range_values);
                }
                catch (...)
                {
                    throw std::runtime_error("Error parsing line " + std::to_string(idx) + "of input file " + param_filename);
                }
                break;
            // #2 foreground probability and associated ranges
            case 2:
                try {
                    fg_prob = std::stod(params[idx][0]);
                    parse_input_range(params[idx][1], fg_range_values);
                }
                catch (...)
                {
                    throw std::runtime_error("Error parsing line " + std::to_string(idx) + "of input file " + param_filename);
                }
                break;
            // #3 real range values
            case 3:
                try {
                    parse_input_range(params[idx][0], range_values);
                }
                catch (...)
                {
                    throw std::runtime_error("Error parsing line " + std::to_string(idx) + "of input file " + param_filename);
                }
                break;
            // #4 camera parameters
            case 4:
                try {
                    pixel_size = std::stod(params[idx][0]);
                    focal_length = std::stod(params[idx][1]);
                    baseline = std::stod(params[idx][2]);
                }
                catch (...)
                {
                    throw std::runtime_error("Error parsing line " + std::to_string(idx) + "of input file " + param_filename);
                }
                break;

            // #5 image size: height, width
            case 5:
                try
                {
                    img_h = (uint32_t)std::stoi(params[idx][0]);
                    img_w = (uint32_t)std::stoi(params[idx][1]);
                }
                catch (...)
                {
                    img_h = 512;
                    img_w = 512;
                }
                break;

            // #6 maximum number of depthmap values within a single image
            case 6:
                try {
                    max_dm_vals_per_image = (uint32_t)std::stoi(params[idx][0]);
                }
                catch (...)
                {
                    throw std::runtime_error("Error parsing line " + std::to_string(idx) + "of input file " + param_filename);
                }
                break;

            // #7 number of images to generate
            case 7:
                try {
                    num_images = (uint32_t)std::stoi(params[idx][0]);
                }
                catch (...)
                {
                    throw std::runtime_error("Error parsing line " + std::to_string(idx) + "of input file " + param_filename);
                }
                break;
            // #8 save location
            case 8:
                save_location = params[idx][0];
                break;
            default:
                break;
        }
    }

    
}   // end of read_blur_params
    


#endif  // _BLUR_PARAMS_H_
