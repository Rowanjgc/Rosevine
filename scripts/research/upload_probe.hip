// R3a.2: private raw samples only. Run under gpu_campaign.py and the GPU lock.
// Build: hipcc --offload-arch=gfx1101 -O3 -Wall -Werror upload_probe.hip -o PROBE
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static double us(Clock::time_point t) {
    return std::chrono::duration<double, std::micro>(Clock::now()-t).count();
}
static void check(hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr,"HIP error: %s\n",hipGetErrorString(e)); std::exit(1); }
}
__global__ void marker(unsigned *p) { if (!threadIdx.x) *p += 1; }

int main(int argc, char **argv) {
    bool events = argc == 1;
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--no-events"))) return 3;
    constexpr size_t bytes = 2816 * sizeof(float);
    check(hipSetDevice(0));
    void *dev=nullptr, *pinned=nullptr;
    check(hipMalloc(&dev, bytes));
    check(hipHostMalloc(&pinned, bytes));
    check(hipMemset(dev, 0, bytes));
    std::vector<unsigned char> page(bytes, 0x5a), got(bytes);
    std::memcpy(pinned,page.data(),bytes);
    hipEvent_t a,b;
    check(hipEventCreate(&a)); check(hipEventCreate(&b));
    std::puts("staging,api,idle_us,pre_marker,sample,marker_host_us,enqueue_us,wait_us,total_us,event_us");
    // Interleave staging/API variants within each repetition, reverse alternate repetitions.
    for (int idle : {0,500,5000,20000}) for (int pre : {0,1}) for (int rep=0;rep<42;++rep) {
        for (int order=0;order<4;++order) {
            int variant = rep%2 ? 3-order : order;
            bool pin = variant & 1, async = variant & 2;
            void *src = pin ? pinned : page.data();
            hipLaunchKernelGGL(marker, dim3(1), dim3(64), 0, 0, static_cast<unsigned*>(dev));
            check(hipGetLastError()); check(hipDeviceSynchronize());
            std::this_thread::sleep_for(std::chrono::microseconds(idle));
            double marker_us = 0;
            if (pre) {
                auto t=Clock::now();
                hipLaunchKernelGGL(marker, dim3(1), dim3(64), 0, 0, static_cast<unsigned*>(dev));
                check(hipGetLastError()); check(hipDeviceSynchronize());
                marker_us=us(t);
            }
            auto total=Clock::now();
            // Event record before the copy can itself wake the device; report this limitation.
            if (events) check(hipEventRecord(a,0));
            auto t=Clock::now();
            if (async) check(hipMemcpyAsync(dev,src,bytes,hipMemcpyHostToDevice,0));
            else check(hipMemcpy(dev,src,bytes,hipMemcpyHostToDevice));
            double enqueue=us(t);
            if (events) check(hipEventRecord(b,0));
            t=Clock::now();
            if (events) check(hipEventSynchronize(b));
            else check(hipDeviceSynchronize());
            double wait=us(t);
            double host_total=us(total); float event_ms=0;
            if (events) check(hipEventElapsedTime(&event_ms,a,b));
            check(hipMemcpy(got.data(),dev,bytes,hipMemcpyDeviceToHost));
            if (std::memcmp(got.data(),src,bytes)) return 2;
            if (rep>=2 && events) std::printf("%s,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                pin?"pinned":"pageable",async?"async":"blocking",idle,pre,rep-2,
                marker_us,enqueue,wait,host_total,event_ms*1000);
            if (rep>=2 && !events) std::printf("%s,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,null\n",
                pin?"pinned":"pageable",async?"async":"blocking",idle,pre,rep-2,
                marker_us,enqueue,wait,host_total);
        }
    }
    check(hipEventDestroy(b)); check(hipEventDestroy(a));
    check(hipHostFree(pinned)); check(hipFree(dev));
}
