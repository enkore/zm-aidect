#!/bin/bash -xue

# Builds a binary linked against system OpenCV

apt-get update
apt-get install -y clang libclang-dev llvm-dev libopencv-dev libopencv-dnn-dev

cargo build --release
mkdir -p artifact
cp target/release/zm-aidect artifact
cp yolov4-tiny.cfg yolov4-tiny.weights artifact

cp zm-aidect@.service artifact/
sed -i "s#ExecStart=#ExecStart=/opt#" artifact/zm-aidect@.service

tee artifact/INSTALL <<EOF

This package is compiled for Debian 11 Bullseye.

- Extact the tarball to a location of your choosing, e.g. /opt/zm-aidect
- Install system dependencies: apt install libopencv-core4.5 libopencv-imgproc4.5 libopencv-dnn4.5
- Adjust the ExecStart line in zm-aidect@.service to match the path you extracted to
  (if you used /opt/zm-aidect, you don't need to change anything)
- Register the systemd service: systemctl enable /opt/zm-aidect/zm-aidect@.service
- Configure aidect zones like the README describes for the monitors you want to use it on
- Enable and start zm-aidect for the monitors, e.g.

     systemctl enable zm-aidect@{1,2,3,4}
     systemctl start zm-aidect@{1,2,3,4}
EOF

