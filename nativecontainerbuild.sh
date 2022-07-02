#!/bin/bash -xue

# Builds a binary linked against system OpenCV

apt update
apt install -y clang libclang-dev llvm-dev libopencv-dev libopencv-dnn-dev

cargo build --release
mkdir -p artifact
cp target/release/zm-aidect artifact

# INSTALL
# apt install libopencv-core4.5 libopencv-imgproc4.5 libopencv-dnn4.5
# extract to e.g. /opt/zm-aidect
# systemctl enable /opt/zm-aidect/zm-aidect@{1,2,3,4}.service
