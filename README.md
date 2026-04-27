# Superpoint ROS2 testing node

test for superpoint running as ros2 node
# build
tested on:
- ros2 jazzy, with mrs_lib
- libtorch 2.9.1. cuda 12.8; on 570 nvidia drivers

`colcon build --cmake-args -DCMAKE_CUDA_ARCHITECTURES=86`
!needs cuda toolkit 
- `export CUDA_HOME=/usr/local/cuda-12.8`
- `export PATH=$CUDA_HOME/bin:$PATH`

disclaimer - had to convert weights from MagicLeap PyTorch implementation to use with libtorch, but it seems it now doesn't give correct descriptors  
