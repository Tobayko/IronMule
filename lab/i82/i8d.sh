T=~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16
cd $T/vendor/llama.cpp
F=ggml/src/ggml-cuda/ggml-cuda.cu
python3 /work/mki8b.py /work/ggml-cuda.cu.pk4state $F && cmake --build build --target llama -j 16 > /work/logs/i82build.log 2>&1 || { grep -m8 -i "error" /work/logs/i82build.log; exit 1; }
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
export IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUBLAS_MIN_ROWS=512 IRONMULE_CUDA_GRAPHS_ROWS=32
cd /work/IronMule
for d in 0 1; do for m in 2 6; do IRONMULE_I8_DIRECT=$d IRONMULE_I8=$m python /work/ppl_i8.py 2>&1 | grep "^PPL" | sed "s/^PPL /PPL {\"direct\": $d, /; s/, {/, /"; done; done
echo I8D-DONE
