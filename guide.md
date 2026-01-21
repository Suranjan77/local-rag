mkdir build && cd build
cmake .. -G Ninja -DGGML_HIPBLAS=ON -DAMDGPU_TARGETS="gfx1103"