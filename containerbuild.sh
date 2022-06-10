#!/bin/bash -xue
apt update
apt install -y clang libclang-dev llvm-dev cmake

cd /
git clone --branch 4.6.0 --depth 1 https://github.com/opencv/opencv
mkdir opencv/build
cd opencv/build
cmake .. -DCPU_BASELINE=SSE4_2 -DCPU_DISPATCH= -DOPENCV_GENERATE_PKGCONFIG=YES -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
  -DWITH_OPENCL=OFF -DWITH_FFMPEG=OFF -DWITH_V4L=OFF -DWITH_GSTREAMER=OFF -DWITH_1394=OFF \
  -DWITH_OPENEXR=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_WEBP=OFF -DWITH_TIFF=OFF -DWITH_JPEG=OFF \
  -DWITH_IMGCODEC_HDR=OFF -DWITH_IMGCODEC_SUNRASTER=OFF -DWITH_IMGCODEC_PXM=OFF -DWITH_IMGCODEC_PFM=OFF \
  -DBUILD_LIST=dnn -DBUILD_WITH_DYNAMIC_IPP=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_OPENGL=OFF
make -j12
make install

cd /code

#export OPENCV_INCLUDE_PATHS=/opencv/install/usr/local/include/opencv4/
#export OPENCV_LINK_PATHS=/opencv/usr/local/lib
#export OPENCV_LINK_LIBS=opencv_core,opencv_dnn,opencv_imgproc
#,ittnotify,libprotobuf,ippip,ippicv,z,va,va-drm,OpenGL,GLX,GLU,dl,m,pthread,rt
# ittnotify -llibprotobuf -lippiw -lippicv -L/lib64 -lz -lva -lva-drm -L/usr/lib -lOpenGL -lGLX -lGLU -ldl -lm -lpthread -lrt
RUSTFLAGS='-C link-args=-Wl,-rpath,$ORIGIN' cargo build --release

mkdir -p artifact
cp target/release/zm-aidect artifact
cp yolov4-tiny.cfg yolov4-tiny.weights artifact
cp /usr/local/lib/libopencv_*.so.406 artifact
