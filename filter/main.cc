#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "filter.hh"
#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1) {
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1-t0).count();
    if (dt == 0) { return 0; }
    return ((n+n+n)*sizeof(float)*1e-9)/(dt*1e-6);
}

void print(const char* name, std::array<duration,5> dt) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    std::cout << '\n';
}

void print_column_names() {
    std::cout << std::setw(19) << "function";
    std::cout << std::setw(20) << "OpenMP";
    std::cout << std::setw(20) << "OpenCL total";
    std::cout << std::setw(20) << "OpenCL copy-in";
    std::cout << std::setw(20) << "OpenCL kernel";
    std::cout << std::setw(20) << "OpenCL copy-out";
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_filter(int n, OpenCL& opencl) {
    auto input = random_std_vector<float>(n);
    int local_sz = 256;
    std::vector<int> bins(n / local_sz);
    std::vector<float> result, result_gpu;
    result.reserve(n);
    result_gpu.reserve(n);
    cl::Kernel cnt_pos_kernel(opencl.program, "count_positive");
    auto t0 = clock_type::now();
    filter(input, result, [] (float x) { return x > 0; }); // filter positive numbers
    auto t1 = clock_type::now();
    cl::Buffer d_input(opencl.queue, begin(input), end(input), false);
    cl::Buffer d_bins(opencl.context, CL_MEM_READ_WRITE, bins.size()*sizeof(int));
    opencl.queue.finish();
    cnt_pos_kernel.setArg(0, d_input);
    cnt_pos_kernel.setArg(1, d_bins);
    opencl.queue.flush();
    auto t2 = clock_type::now();
    opencl.queue.enqueueNDRangeKernel(cnt_pos_kernel, cl::NullRange, cl::NDRange(n), cl::NDRange(local_sz));
    opencl.queue.flush();
    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_bins, std::begin(bins), std::end(bins));
    auto t4 = clock_type::now();
    verify_vector(result, result_gpu);
    print("filter", {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3});
}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_filter(1024*1024, opencl);
}

const std::string src = R"(
#define BUFFSIZE 1024
kernel void filter(global const float *input,
                    global int *res_size,
                    global float *result) {
    const int i = get_global_id(0);
    const int n = get_global_size(0);
    int t = get_local_id(0);
    if (i == 0)
        res_size[0] = n;
}

kernel void count_positive(global const float *a,
                    global int *result) {
    const int m = get_local_size(0);
    int k = get_group_id(0);
    int t = get_local_id(0);
    local float buff[BUFFSIZE];

    buff[t] = a[k * m + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (t == 0) {
        int cnt = 0;
        for (int j = 0; j < m; j++) {
            if (buff[j] > 0)
                cnt++;
        }
        result[k] = cnt;
    }
}
)";

int main() {
    try {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
            return 1;
        }
        cl::Platform platform = platforms[0];
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        cl::Program program(context, src);
        // compile the programme
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            for (const auto& device : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        cl::CommandQueue queue(context, device);
        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
            << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return 1;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
