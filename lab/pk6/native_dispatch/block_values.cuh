// GGUF Q4_K and Q6_K dequantization to half. Source weights are immutable.
#pragma once
template<int Type>
__global__ void half_weights(const unsigned char *raw, half *out, int64_t blocks) {
    int64_t block = blockIdx.x;
    if (block >= blocks) return;
    int i = threadIdx.x;
    const unsigned char *p = raw + block * (Type == 12 ? 144 : 210);
    float value;
    if constexpr (Type == 12) {
        int g = i / 32;
        const auto *s = p + 4;
        int scale = g < 4 ? s[g]&63 : (s[g+4]&15) | ((s[g-4] >> 6) << 4);
        int minimum = g < 4 ? s[g+4]&63 : (s[g+4] >> 4) | ((s[g] >> 6) << 4);
        int q = p[16 + (g / 2)*32 + i%32];
        int code = (g & 1) ? q >> 4 : q & 15;
        value = __half2float(*reinterpret_cast<const half *>(p)) * scale * code -
                __half2float(*reinterpret_cast<const half *>(p+2)) * minimum;
    } else {
        int segment = i / 128, g = i%128 / 32, off = i%32;
        int ql = p[segment*64 + (g&1)*32 + off];
        int qh = p[128 + segment*32 + off];
        int code = ((g < 2 ? ql & 15 : ql >> 4) | (((qh >> (2*g)) & 3) << 4)) - 32;
        value = __half2float(*reinterpret_cast<const half *>(p+208)) *
                reinterpret_cast<const int8_t *>(p+192)[i/16] * code;
    }
    out[block*256 + i] = __float2half_rn(value);
}
