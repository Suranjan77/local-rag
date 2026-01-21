cmake -B build -G Ninja \
    -DGGML_HIPBLAS=ON \
    -DCMAKE_HIP_ARCHITECTURES="gfx1151" \
    -DCMAKE_C_COMPILER=/opt/rocm/bin/hipcc \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
    -DCMAKE_BUILD_TYPE=Release
