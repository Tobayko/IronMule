set -e
T=~/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16
cd $T/vendor/llama.cpp
F=ggml/src/ggml-cuda/ggml-cuda.cu
test -f /work/ggml-cuda.cu.pk4state || cp $F /work/ggml-cuda.cu.pk4state
python3 /work/mki8b.py /work/ggml-cuda.cu.pk4state /work/ggml-cuda.cu.i82
cp /work/ggml-cuda.cu.i82 $F
cmake --build build --target llama -j 16 > /work/logs/i82build.log 2>&1 || { grep -m8 -i "error" /work/logs/i82build.log; exit 1; }
tail -1 /work/logs/i82build.log
set +e
export IRONMULE_GGML_BINDINGS=$T LLAMA_CPP_LIB_PATH=$T/vendor/llama.cpp/build/bin
export IRONMULE_GGML_PACK=1 IRONMULE_GGML_UBATCH=8192 IRONMULE_CUBLAS_MIN_ROWS=512 IRONMULE_CUDA_GRAPHS_ROWS=32
cd /work/IronMule
IRONMULE_PK5=0 python /work/ppl_i8b.py 0 2>&1 | grep "^PPL\|rror" | tail -3
python /work/ppl_i8b.py 0,3,4,2,5,7,6 2>&1 | grep "^PPL\|rror" | tail -9
IRONMULE_I8_MAX_K=16384 python /work/ppl_i8b.py 5,7,6 2>&1 | grep "^PPL\|rror" | tail -4
echo I82-DONE
