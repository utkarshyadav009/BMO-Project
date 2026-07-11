// nvmap_alloc_probe.cu — standalone probe to certify (or kill) the
// per-cudaMalloc NvMap allocation overhead hypothesis. Not part of the
// personaplex build; no model code touched. Measurement only.
//
// Compares three allocation patterns using the exact same total bytes
// (the 31-layer BMO tensor total, ~1826 MiB):
//   Pattern A: 62 separate cudaMalloc calls (one per real BMO tensor size)
//   Pattern B: 8 equal-sized slabs
//   Pattern C: 1 single slab
// If per-alloc overhead is large for A but small for B/C, that's direct
// evidence the 62-separate-buffers design (not host memory, not payload
// content) is the dominant unaccounted consumer.
//
// Build: nvcc -o nvmap_alloc_probe nvmap_alloc_probe.cu -arch=sm_87
// Run:   sudo ./nvmap_alloc_probe   (sudo so the nvmap debug node is readable)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

static size_t free_mib() {
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    return free_b / 1024 / 1024;
}

static void dump_nvmap_debug(const char * label) {
    FILE * f = fopen("/sys/kernel/debug/nvmap/iovmm/clients", "r");
    if (!f) {
        printf("NVMAP_DEBUG[%s]: unavailable (not root-readable)\n", label);
        return;
    }
    printf("NVMAP_DEBUG[%s]:\n", label);
    char line[512];
    int n = 0;
    while (fgets(line, sizeof(line), f) && n < 40) {
        printf("  %s", line);
        n++;
    }
    fclose(f);
}

struct PatternResult {
    const char * name;
    size_t requested_total_B;
    size_t delta_total_MiB;
    double overhead_total_MiB;
    double overhead_per_alloc_MiB;
    size_t residual_after_free_MiB;
    bool single_alloc_succeeded; // only meaningful for pattern C
};

// Real per-tensor BMO alloc sizes, alternating in/out, 31 layers (62 total),
// taken directly from the MEMLEDGER ledger (alloc_B field).
static const size_t IN_WEIGHT_BYTES  = 39279104;
static const size_t OUT_WEIGHT_BYTES = 19639296;

static PatternResult run_pattern_A() {
    size_t before = free_mib();
    std::vector<void*> ptrs;
    size_t requested = 0;
    for (int layer = 0; layer < 31; layer++) {
        void * p1 = nullptr, * p2 = nullptr;
        cudaError_t e1 = cudaMalloc(&p1, IN_WEIGHT_BYTES);
        cudaError_t e2 = cudaMalloc(&p2, OUT_WEIGHT_BYTES);
        if (e1 == cudaSuccess) { ptrs.push_back(p1); requested += IN_WEIGHT_BYTES; }
        if (e2 == cudaSuccess) { ptrs.push_back(p2); requested += OUT_WEIGHT_BYTES; }
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            fprintf(stderr, "PATTERN A: alloc failed at layer %d (in=%s out=%s) — stopping early\n",
                    layer, cudaGetErrorString(e1), cudaGetErrorString(e2));
            break;
        }
    }
    size_t after = free_mib();
    dump_nvmap_debug("A_peak");
    size_t delta_mib = before - after;
    double overhead_mib = (double)delta_mib - (double)requested / 1024.0 / 1024.0;
    for (void * p : ptrs) CUDA_CHECK(cudaFree(p));
    size_t after_free = free_mib();
    size_t residual = before - after_free; // positive = didn't fully return
    return { "A (62 separate)", requested, delta_mib, overhead_mib,
             overhead_mib / (double)ptrs.size(), residual, false };
}

static PatternResult run_pattern_B() {
    size_t total = 31 * (IN_WEIGHT_BYTES + OUT_WEIGHT_BYTES);
    size_t slab = total / 8;
    size_t before = free_mib();
    std::vector<void*> ptrs;
    size_t requested = 0;
    for (int i = 0; i < 8; i++) {
        void * p = nullptr;
        cudaError_t e = cudaMalloc(&p, slab);
        if (e == cudaSuccess) { ptrs.push_back(p); requested += slab; }
        else {
            fprintf(stderr, "PATTERN B: alloc failed at slab %d: %s — stopping early\n", i, cudaGetErrorString(e));
            break;
        }
    }
    size_t after = free_mib();
    dump_nvmap_debug("B_peak");
    size_t delta_mib = before - after;
    double overhead_mib = (double)delta_mib - (double)requested / 1024.0 / 1024.0;
    for (void * p : ptrs) CUDA_CHECK(cudaFree(p));
    size_t after_free = free_mib();
    size_t residual = before - after_free;
    return { "B (8 slabs)", requested, delta_mib, overhead_mib,
             overhead_mib / (double)ptrs.size(), residual, false };
}

static PatternResult run_pattern_C() {
    size_t total = 31 * (IN_WEIGHT_BYTES + OUT_WEIGHT_BYTES);
    size_t before = free_mib();
    void * p = nullptr;
    cudaError_t e = cudaMalloc(&p, total);
    bool ok = (e == cudaSuccess);
    if (!ok) {
        fprintf(stderr, "PATTERN C: single %.2f MiB alloc FAILED: %s\n", total / 1024.0 / 1024.0, cudaGetErrorString(e));
    }
    size_t after = free_mib();
    dump_nvmap_debug("C_peak");
    size_t delta_mib = before - after;
    double overhead_mib = ok ? ((double)delta_mib - (double)total / 1024.0 / 1024.0) : 0.0;
    if (ok) CUDA_CHECK(cudaFree(p));
    size_t after_free = free_mib();
    size_t residual = before - after_free;
    return { "C (1 slab)", ok ? total : 0, delta_mib, overhead_mib, overhead_mib, residual, ok };
}

int main() {
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    printf("BASELINE: free_MiB=%zu total_MiB=%zu\n", free_b/1024/1024, total_b/1024/1024);
    dump_nvmap_debug("baseline");

    PatternResult a = run_pattern_A();
    PatternResult b = run_pattern_B();
    PatternResult c = run_pattern_C();

    printf("\n=== RESULT TABLE ===\n");
    printf("%-16s %14s %14s %16s %20s %16s %10s\n",
           "pattern", "requested_MiB", "delta_MiB", "overhead_MiB", "overhead_per_alloc_MiB", "residual_MiB", "c_ok");
    for (PatternResult * r : {&a, &b, &c}) {
        printf("%-16s %14.2f %14zu %16.2f %20.4f %16zu %10s\n",
               r->name, r->requested_total_B/1024.0/1024.0, r->delta_total_MiB,
               r->overhead_total_MiB, r->overhead_per_alloc_MiB, r->residual_after_free_MiB,
               r->single_alloc_succeeded ? "yes" : "n/a");
    }
    return 0;
}
