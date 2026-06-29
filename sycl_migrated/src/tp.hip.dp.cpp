/*
 * tp.hip — Tensor Parallelism communication for 2-GPU TP
 *
 * Host-staged AllReduce for PCIe-connected GPUs (no NVLink).
 * For 12KB payloads (n_embd=3072), host staging is simpler and
 * competitive with P2P since PCIe round-trip dominates either way.
 */

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "tp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" tp_ctx_t *tp_init(int max_reduce_size) try {
    int device_count = 0;
    HIP_CHECK(DPCT_CHECK_ERROR(device_count = dpct::device_count()));
    if (device_count < 2) {
        fprintf(stderr, "TP: need 2 GPUs, found %d\n", device_count);
        return nullptr;
    }

    tp_ctx_t *ctx = (tp_ctx_t *)calloc(1, sizeof(tp_ctx_t));
    if (!ctx) return nullptr;

    ctx->n_devices = 2;
    ctx->device_ids[0] = 0;
    ctx->device_ids[1] = 1;
    ctx->max_reduce_size = max_reduce_size;

    // Print device info
    for (int d = 0; d < 2; d++) {
        hipDeviceProp_t props;
        HIP_CHECK(DPCT_CHECK_ERROR(dpct::get_device(d).get_device_info(props)));
        fprintf(stderr, "TP: GPU%d: %s (%zu MB VRAM, %d CUs)\n", d,
                props.get_name(), props.get_global_mem_size() / (1024 * 1024),
                props.get_max_compute_units());
    }

    // Create streams and events on each device
    for (int d = 0; d < 2; d++) {
        /*
        DPCT1093:79: The "d" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        HIP_CHECK(dpct::select_device(d));
        HIP_CHECK(DPCT_CHECK_ERROR(
            ctx->compute[d] = dpct::get_current_device().create_queue()));
        HIP_CHECK(DPCT_CHECK_ERROR(
            ctx->transfer[d] = dpct::get_current_device().create_queue()));
        HIP_CHECK(DPCT_CHECK_ERROR(ctx->reduce_ready[d] = new sycl::event()));
        HIP_CHECK(DPCT_CHECK_ERROR(ctx->reduce_done[d] = new sycl::event()));

        // Scratch buffer for AllReduce result on each GPU
        HIP_CHECK(
            DPCT_CHECK_ERROR(ctx->d_reduced[d] = sycl::malloc_device<float>(
                                 max_reduce_size, dpct::get_in_order_queue())));
    }

    // Allocate pinned host buffers for staging
    /*
    DPCT1093:80: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
    HIP_CHECK(hipHostMalloc(&ctx->h_partial[0], max_reduce_size * sizeof(float)));
    HIP_CHECK(hipHostMalloc(&ctx->h_partial[1], max_reduce_size * sizeof(float)));
    HIP_CHECK(hipHostMalloc(&ctx->h_reduced, max_reduce_size * sizeof(float)));

    /*
    DPCT1093:81: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));

    fprintf(stderr, "TP: initialized 2-GPU TP, max_reduce_size=%d (%.1f KB)\n",
            max_reduce_size, max_reduce_size * sizeof(float) / 1024.0f);

    return ctx;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

extern "C" void tp_free(tp_ctx_t *ctx) try {
    if (!ctx) return;

    for (int d = 0; d < 2; d++) {
        /*
        DPCT1093:82: The "d" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        HIP_CHECK(dpct::select_device(d));
        dpct::get_current_device().destroy_queue(ctx->compute[d]);
        dpct::get_current_device().destroy_queue(ctx->transfer[d]);
        dpct::destroy_event(ctx->reduce_ready[d]);
        dpct::destroy_event(ctx->reduce_done[d]);
        dpct::dpct_free(ctx->d_reduced[d], dpct::get_in_order_queue());
    }

    /*
    DPCT1093:83: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
    sycl::free(ctx->h_partial[0], dpct::get_in_order_queue());
    sycl::free(ctx->h_partial[1], dpct::get_in_order_queue());
    sycl::free(ctx->h_reduced, dpct::get_in_order_queue());

    free(ctx);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

extern "C" void tp_allreduce_sum(tp_ctx_t *ctx, float *d_partial0,
                                 float *d_partial1, float *d_out0,
                                 float *d_out1, int n) try {
    // Step 1: D2H both partials (in parallel on separate streams)
    /*
    DPCT1093:84: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
    /*
    DPCT1124:85: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->compute[0]->memcpy(
        ctx->h_partial[0], d_partial0, n * sizeof(float))));
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->compute[0]->wait()));

    /*
    DPCT1093:86: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(1));
    /*
    DPCT1124:87: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->compute[1]->memcpy(
        ctx->h_partial[1], d_partial1, n * sizeof(float))));
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->compute[1]->wait()));

    // Step 2: CPU reduce
    for (int i = 0; i < n; i++) {
        ctx->h_reduced[i] = ctx->h_partial[0][i] + ctx->h_partial[1][i];
    }

    // Step 3: H2D reduced result to both GPUs
    /*
    DPCT1093:88: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
    /*
    DPCT1124:89: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    HIP_CHECK(DPCT_CHECK_ERROR(
        ctx->compute[0]->memcpy(d_out0, ctx->h_reduced, n * sizeof(float))));

    /*
    DPCT1093:90: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(1));
    /*
    DPCT1124:91: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    HIP_CHECK(DPCT_CHECK_ERROR(
        ctx->compute[1]->memcpy(d_out1, ctx->h_reduced, n * sizeof(float))));

    // Sync both
    /*
    DPCT1093:92: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->compute[0]->wait()));
    /*
    DPCT1093:93: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(1));
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->compute[1]->wait()));

    /*
    DPCT1093:94: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

extern "C" void tp_allreduce_sum_async(tp_ctx_t *ctx, float *d_partial0,
                                       float *d_partial1, float *d_out0,
                                       float *d_out1, int n) try {
    // D2H on transfer streams (non-blocking on compute)
    /*
    DPCT1093:95: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
    /*
    DPCT1024:96: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    HIP_CHECK(DPCT_CHECK_ERROR(
        dpct::sync_barrier(ctx->reduce_ready[0], ctx->compute[0])));
    HIP_CHECK(DPCT_CHECK_ERROR(
        ctx->transfer[0]->ext_oneapi_submit_barrier({*ctx->reduce_ready[0]})));
    /*
    DPCT1124:97: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->transfer[0]->memcpy(
        ctx->h_partial[0], d_partial0, n * sizeof(float))));

    /*
    DPCT1093:98: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(1));
    /*
    DPCT1024:99: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    HIP_CHECK(DPCT_CHECK_ERROR(
        dpct::sync_barrier(ctx->reduce_ready[1], ctx->compute[1])));
    HIP_CHECK(DPCT_CHECK_ERROR(
        ctx->transfer[1]->ext_oneapi_submit_barrier({*ctx->reduce_ready[1]})));
    /*
    DPCT1124:100: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->transfer[1]->memcpy(
        ctx->h_partial[1], d_partial1, n * sizeof(float))));

    // Store params for wait
    // (We store d_out pointers in reduced buffers temporarily — will need them in wait)
    // Actually, just use the sync version internally for now since the payload is tiny
    /*
    DPCT1093:101: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->transfer[0]->wait()));
    /*
    DPCT1093:102: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(1));
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->transfer[1]->wait()));

    // CPU reduce
    for (int i = 0; i < n; i++) {
        ctx->h_reduced[i] = ctx->h_partial[0][i] + ctx->h_partial[1][i];
    }

    // H2D to both
    /*
    DPCT1093:103: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
    /*
    DPCT1124:104: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    HIP_CHECK(DPCT_CHECK_ERROR(
        ctx->compute[0]->memcpy(d_out0, ctx->h_reduced, n * sizeof(float))));
    /*
    DPCT1093:105: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(1));
    /*
    DPCT1124:106: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    HIP_CHECK(DPCT_CHECK_ERROR(
        ctx->compute[1]->memcpy(d_out1, ctx->h_reduced, n * sizeof(float))));

    /*
    DPCT1093:107: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

extern "C" void tp_allreduce_wait(tp_ctx_t *ctx) try {
    /*
    DPCT1093:108: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->compute[0]->wait()));
    /*
    DPCT1093:109: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(1));
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->compute[1]->wait()));
    /*
    DPCT1093:110: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

extern "C" void tp_broadcast(tp_ctx_t *ctx, int src_rank, int dst_rank,
                             const void *d_src, void *d_dst, size_t bytes) try {
    // Host-staged copy: src GPU → pinned host → dst GPU
    /*
    DPCT1093:111: The "src_rank" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    HIP_CHECK(dpct::select_device(src_rank));
    /*
    DPCT1124:112: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    HIP_CHECK(DPCT_CHECK_ERROR(
        ctx->transfer[src_rank]->memcpy(ctx->h_partial[0], d_src, bytes)));
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->transfer[src_rank]->wait()));

    /*
    DPCT1093:113: The "dst_rank" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    HIP_CHECK(dpct::select_device(dst_rank));
    /*
    DPCT1124:114: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    HIP_CHECK(DPCT_CHECK_ERROR(
        ctx->compute[dst_rank]->memcpy(d_dst, ctx->h_partial[0], bytes)));
    HIP_CHECK(DPCT_CHECK_ERROR(ctx->compute[dst_rank]->wait()));

    /*
    DPCT1093:115: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    HIP_CHECK(dpct::select_device(0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
