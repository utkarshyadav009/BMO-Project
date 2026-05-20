// Deliverable 2: compare fused_dequant_matvec_proto (production device pointers
// from bmo_prepare_device_packed_tensors) against PyTorch-generated golden y.

#include "bmo.h"
#include "bmo_proto_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

static const char kMagicStr[4] = { 'B', 'M', 'V', '5' };

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (expr);                                                       \
        if (_err != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << cudaGetErrorString(_err) << " at " << __FILE__ \
                      << ":" << __LINE__ << " (" << #expr << ")\n";                      \
            std::exit(2);                                                                \
        }                                                                                \
    } while (0)

struct GoldenTensor {
    std::string name;
    int32_t     rows = 0;
    int32_t     cols = 0;
    std::vector<float> x;
    std::vector<float> y_gt;
};

static bool read_golden(const char *path, std::vector<GoldenTensor> *out, std::string *err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        *err = std::string("cannot open ") + path;
        return false;
    }
    char magic[4];
    f.read(magic, 4);
    if (!f || std::memcmp(magic, kMagicStr, 4) != 0) {
        *err = "bad magic (expected BMV5)";
        return false;
    }
    uint32_t ver = 0, n = 0;
    f.read(reinterpret_cast<char *>(&ver), sizeof(ver));
    f.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (ver != 1 || n == 0 || n > 100) {
        *err = "bad header version/n_tensors";
        return false;
    }
    out->clear();
    out->resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        GoldenTensor &t = (*out)[i];
        uint32_t name_len = 0;
        f.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
        if (name_len == 0 || name_len > 1u << 20) {
            *err = "bad name_len";
            return false;
        }
        t.name.resize(name_len);
        f.read(t.name.data(), name_len);
        f.read(reinterpret_cast<char *>(&t.rows), sizeof(t.rows));
        f.read(reinterpret_cast<char *>(&t.cols), sizeof(t.cols));
        if (t.rows <= 0 || t.cols <= 0) {
            *err = "bad rows/cols";
            return false;
        }
        uint32_t nx = 0;
        f.read(reinterpret_cast<char *>(&nx), sizeof(nx));
        if (nx != (uint32_t) t.cols) {
            *err = "x length mismatch";
            return false;
        }
        t.x.resize((size_t) t.cols);
        t.y_gt.resize((size_t) t.rows);
        f.read(reinterpret_cast<char *>(t.x.data()), (std::streamsize)(sizeof(float) * (size_t) t.cols));
        f.read(reinterpret_cast<char *>(t.y_gt.data()), (std::streamsize)(sizeof(float) * (size_t) t.rows));
        if (!f) {
            *err = "short read";
            return false;
        }
    }
    return true;
}

static double cosine_vec(const std::vector<float> &a, const std::vector<float> &b) {
    double dot = 0, na = 0, nb = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        dot += (double) a[i] * (double) b[i];
        na += (double) a[i] * (double) a[i];
        nb += (double) b[i] * (double) b[i];
    }
    const double dn = std::sqrt(na) * std::sqrt(nb);
    return dn > 0 ? dot / dn : 0.0;
}

static float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
    float m = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static void print_top5(
    const std::vector<float> &expv, const std::vector<float> &actv) {
    struct Item {
        size_t i;
        float  d;
    };
    std::vector<Item> items;
    const size_t n = std::min(expv.size(), actv.size());
    items.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        items.push_back(Item { i, std::fabs(expv[i] - actv[i]) });
    }
    std::partial_sort(
        items.begin(),
        items.begin() + (std::min)(size_t(5), items.size()),
        items.end(),
        [](const Item &a, const Item &b) { return a.d > b.d; });
    const size_t k = std::min(size_t(5), items.size());
    std::cerr << "  top-" << k << " row discrepancies (row, expected, actual, |diff|):\n";
    for (size_t j = 0; j < k; ++j) {
        const size_t i = items[j].i;
        std::cerr << "    row " << i << " exp=" << expv[i] << " act=" << actv[i] << " |d|=" << items[j].d
                  << "\n";
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <golden.bin> <weights_v5.gguf>\n"
                  << "  Generate golden.bin with: python scripts/gen_v5_matvec_golden.py ...\n";
        return 1;
    }

#ifndef BMO_JETSON
    std::cerr << "[bmo_v5_runtime_test] SKIP: built without BMO_JETSON (need Jetson fused registry layout)\n";
    return 0;
#endif

#ifdef BMO_JETSON
    std::vector<GoldenTensor> gold;
    std::string              gerr;
    if (!read_golden(argv[1], &gold, &gerr)) {
        std::cerr << "golden read failed: " << gerr << "\n";
        return 2;
    }

    bmo_model  model;
    bmo_context ctx;
    try {
        bmo_load_model(argv[2], model, ctx);
    } catch (const std::exception &ex) {
        std::cerr << "bmo_load_model failed: " << ex.what() << "\n";
        return 3;
    }

    bool all_ok = true;
    for (GoldenTensor &t : gold) {
        auto it = ctx.packed_registry.find(t.name);
        if (it == ctx.packed_registry.end()) {
            std::cerr << "FAIL: tensor not in packed_registry: " << t.name << "\n";
            all_ok = false;
            continue;
        }
        device_packed_t &dp = it->second;
        if (!dp.is_valid || !dp.preloaded) {
            std::cerr << "FAIL: invalid or not preloaded: " << t.name << "\n";
            all_ok = false;
            continue;
        }
        if (dp.packing_version < 5) {
            std::cerr << "FAIL: expected packing_version>=5 for " << t.name << " got " << dp.packing_version << "\n";
            all_ok = false;
            continue;
        }
        if (dp.rows != t.rows || dp.cols != t.cols) {
            std::cerr << "FAIL: shape mismatch " << t.name << " gguf rows=" << dp.rows << " cols=" << dp.cols
                      << " golden rows=" << t.rows << " cols=" << t.cols << "\n";
            all_ok = false;
            continue;
        }
        if (!dp.row_c2 || !dp.row_c4 || !dp.row_c8 || !dp.row_c16) {
            std::cerr << "FAIL: missing row tier tables for " << t.name << "\n";
            all_ok = false;
            continue;
        }

        const void *kern_pw = dp.canonical_pw_dev;
        const void *kern_pm = dp.canonical_pm_dev;
        const void *kern_fv = dp.canonical_fv_dev;
        if (!kern_pw || !kern_pm) {
            std::cerr << "FAIL: null device pointers for " << t.name << "\n";
            all_ok = false;
            continue;
        }

        const int    block_size_eff = dp.block_size > 0 ? dp.block_size : 32;
        const size_t x_bytes        = (size_t) t.cols * sizeof(float);
        const size_t y_bytes        = (size_t) t.rows * sizeof(float);

        float *d_x = nullptr;
        float *d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
        CUDA_CHECK(cudaMalloc(&d_y, y_bytes));
        CUDA_CHECK(cudaMemcpy(d_x, t.x.data(), x_bytes, cudaMemcpyHostToDevice));

        launch_fused_dequant_matvec_proto(
            kern_pw,
            kern_pm,
            kern_fv,
            dp.row_c2,
            dp.row_c4,
            dp.row_c8,
            dp.row_c16,
            t.rows,
            t.cols,
            block_size_eff,
            dp.n_2bit_bytes,
            dp.n_4bit_bytes,
            dp.scale_low,
            dp.scale_int4,
            dp.scale_int8,
            dp.zp_low,
            dp.zp_int4,
            dp.zp_int8,
            d_x,
            d_y,
            8,
            (cudaStream_t) 0);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        std::vector<float> y_act((size_t) t.rows);
        CUDA_CHECK(cudaMemcpy(y_act.data(), d_y, y_bytes, cudaMemcpyDeviceToHost));
        cudaFree(d_x);
        cudaFree(d_y);

        const double cos   = cosine_vec(t.y_gt, y_act);
        const float  mad   = max_abs_diff(t.y_gt, y_act);
        const bool   pass  = cos >= 0.9999 && mad < 1e-2f;

        std::cout << t.name << "  cos=" << cos << "  max_abs_diff=" << mad << "  " << (pass ? "PASS" : "FAIL")
                  << "\n";

        if (!pass) {
            all_ok = false;
            print_top5(t.y_gt, y_act);
        }
    }

    return all_ok ? 0 : 9;
#endif
}
