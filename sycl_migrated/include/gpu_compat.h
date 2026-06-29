/*
 * gpu_compat.h — Thin compatibility layer for HIP <-> CUDA
 *
 * All GPU code is written using HIP API names (hip*).
 * When building for NVIDIA (FR_CUDA), this header maps them to CUDA equivalents.
 * When building for AMD (default), it just includes the native HIP headers.
 */

#ifndef FR_GPU_COMPAT_H
#define FR_GPU_COMPAT_H

#if defined(FR_CUDA) || defined(FR_SYCL)
// ============================================================
//  NVIDIA CUDA backend
// ============================================================
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

// --- Types ---
#define hipStream_t dpct::queue_ptr
#define hipEvent_t dpct::event_ptr
#define hipError_t dpct::err0
#define hipDeviceProp_t dpct::device_info

// --- Constants ---
#define hipSuccess 0
#define hipMemcpyHostToDevice dpct::host_to_device
#define hipMemcpyDeviceToHost dpct::device_to_host
#define hipMemcpyDeviceToDevice dpct::device_to_device
/*
DPCT1048:0: The original value cudaHostAllocDefault is not meaningful in the
migrated code and was removed or replaced with 0. You may need to check the
migrated code.
*/
#define hipHostMallocDefault 0
#define hipHostRegisterDefault  cudaHostRegisterDefault

// --- Device management ---
#define hipGetDeviceCount       cudaGetDeviceCount
/*
DPCT1093:136: The "0" device may be not the one intended for use. Adjust the
selected device if needed.
*/
/*
DPCT1093:138: The "1" device may be not the one intended for use. Adjust the
selected device if needed.
*/
/*
DPCT1093:140: The "d" device may be not the one intended for use. Adjust the
selected device if needed.
*/
/*
DPCT1093:142: The "i & 1" device may be not the one intended for use. Adjust the
selected device if needed.
*/
/*
DPCT1093:143: The "ctx->gpu1.device_id" device may be not the one intended for
use. Adjust the selected device if needed.
*/
/*
DPCT1093:144: The "last_d" device may be not the one intended for use. Adjust
the selected device if needed.
*/
/*
DPCT1093:169: The "next_d" device may be not the one intended for use. Adjust
the selected device if needed.
*/
/*
DPCT1093:189: The "ctx->dev[0].device_id" device may be not the one intended for
use. Adjust the selected device if needed.
*/
/*
DPCT1093:190: The "dc->device_id" device may be not the one intended for use.
Adjust the selected device if needed.
*/
#define hipSetDevice(d)         DPCT_CHECK_ERROR(dpct::select_device(d))
#define hipGetDeviceProperties  cudaGetDeviceProperties
/*
DPCT1009:78: SYCL reports errors using exceptions and does not use error codes.
Please replace the "get_error_string_dummy(...)" with a real error-handling
function.
*/
#define hipGetErrorString dpct::get_error_string_dummy
/*
DPCT1010:148: SYCL uses exceptions to report errors and does not use the error
codes. The cudaGetLastError function call was replaced with 0. You need to
rewrite this code.
*/
#define hipGetLastError cudaGetLastError
#define hipDeviceSynchronize    cudaDeviceSynchronize

// --- Stream management ---
#define hipStreamCreate         cudaStreamCreate
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamDestroy        cudaStreamDestroy
#define hipStreamSynchronize    cudaStreamSynchronize
#define hipStreamWaitEvent      cudaStreamWaitEvent
#define hipStreamNonBlocking    cudaStreamNonBlocking

// --- Event management ---
#define hipEventCreate          cudaEventCreate
#define hipEventCreateWithFlags cudaEventCreateWithFlags
#define hipEventDestroy         cudaEventDestroy
/*
DPCT1024:154: The original code returned the error code that was further
consumed by the program logic. This original code was replaced with 0. You may
need to rewrite the program logic consuming the error code.
*/
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize     cudaEventSynchronize
#define hipEventElapsedTime     cudaEventElapsedTime
#define hipEventDisableTiming   cudaEventDisableTiming

// --- Device memory ---
#define hipMalloc               cudaMalloc
#define hipFree                 cudaFree
#define hipMemcpy               cudaMemcpy
#define hipMemcpyAsync          cudaMemcpyAsync
#define hipMemset               cudaMemset
#define hipMemsetAsync          cudaMemsetAsync
/*
DPCT1106:141: 'cudaMemGetInfo' was migrated with the Intel extensions for device
information which may not be supported by all compilers or runtimes. You may
need to adjust the code.
*/
#define hipMemGetInfo cudaMemGetInfo

// --- Host/pinned memory ---
template <typename T>
static inline dpct::err0 _fr_hipHostMalloc(T **ptr, size_t size,
                                           unsigned int flags = 0) try {
    return DPCT_CHECK_ERROR(
        *ptr = (T *)sycl::malloc_host(size, dpct::get_in_order_queue()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#define hipHostMalloc           _fr_hipHostMalloc
#define hipHostFree             cudaFreeHost
/*
DPCT1027:146: The call to cudaHostRegister was replaced with 0 because SYCL
currently does not support registering of existing host memory for use by
device. Use USM to allocate memory for use by host and device.
*/
#define hipHostRegister cudaHostRegister
/*
DPCT1026:147: The call to cudaHostUnregister was removed because SYCL currently
does not support registering of existing host memory for use by device. Use USM
to allocate memory for use by host and device.
*/
/*
DPCT1027:191: The call to cudaHostUnregister was replaced with 0 because SYCL
currently does not support registering of existing host memory for use by
device. Use USM to allocate memory for use by host and device.
*/
#define hipHostUnregister cudaHostUnregister

// --- Warp shuffles ---
/*
DPCT1064:1: Migrated __shfl_down_sync call is used in a macro/template
definition and may not be valid for all macro/template uses. Adjust the code.
*/
#define __shfl_down(val, offset)                                               \
    dpct::shift_sub_group_left(                                                \
        sycl::ext::oneapi::this_work_item::get_sub_group(), (val), (offset))
#define __shfl_xor(val, offset)                                                \
    dpct::permute_sub_group_by_xor(                                            \
        sycl::ext::oneapi::this_work_item::get_sub_group(), (val), (offset))
#define __shfl(val, src)                                                       \
    dpct::select_from_sub_group(                                               \
        sycl::ext::oneapi::this_work_item::get_sub_group(), (val), (src))

// --- Kernel attributes ---
/*
DPCT1026:145: The call to cudaFuncSetAttribute was removed because SYCL
currently does not support corresponding setting.
*/
#define hipFuncSetAttribute cudaFuncSetAttribute
#define hipFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize

#else
// ============================================================
//  AMD ROCm / HIP backend (native)
// ============================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#endif // FR_CUDA

// ============================================================
//  Common helpers
// ============================================================

#define HIP_CHECK(call) do {                                                   \
        hipError_t err = (call);                                               \
                                                                               \
    } while (0)

#define HIP_CHECK_I(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#endif // FR_GPU_COMPAT_H
