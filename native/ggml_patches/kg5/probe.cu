// Measurement only: interpose the CUDA backend's normal flash-attention call.
// The builder requires the pinned headers and a dynamic call in the cached DSO.
#include "fattn.cuh"
#include "kg5_dispatch.cuh"
#include <dlfcn.h>

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
    const char *enabled = std::getenv("IRONMULE_KG5");
    if (enabled && std::strcmp(enabled, "1") == 0) {
        ggml_cuda_set_device(ctx.device);
        const int cc = ggml_cuda_info().devices[ctx.device].cc;
        if (kg5_try(ctx, dst, cc)) return;
    }
    static const auto stock = reinterpret_cast<decltype(&ggml_cuda_flash_attn_ext)>(
        dlsym(RTLD_NEXT, "_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor"));
    if (!stock) GGML_ABORT("KG5: missing stock attention in the KG2 library");
    stock(ctx, dst);
}
