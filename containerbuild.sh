#!/bin/bash -xue
apt update
apt install -y clang libclang-dev llvm-dev
export OPENCV_INCLUDE_PATHS=/opencv/usr/local/include/opencv4/
export OPENCV_LINK_PATHS=/opencv/usr/local/lib
export OPENCV_LINK_LIBS=opencv_core,opencv_dnn,opencv_imgproc
cargo build --release
