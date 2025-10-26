#!/bin/bash

# Modify these paths and GPU ids
DATA_PATH="/data/real"
CODE_PATH=".."
GPU_ID="0"

ENVS="  --env=NVIDIA_VISIBLE_DEVICES=$GPU_ID
        --env=CUDA_VISIBLE_DEVICES=$GPU_ID
        --env=NVIDIA_DRIVER_CAPABILITIES=all
        --env=NVIDIA_DISABLE_REQUIRE=1"

VOLUMES="       --volume=$DATA_PATH:/MS3D/dataset"

# Setup environmetns for pop-up visualization of point cloud (open3d)
VISUAL="        --env=DISPLAY
                --env=QT_X11_NO_MITSHM=1
                --volume=/tmp/.X11-unix:/tmp/.X11-unix"
xhost +local:docker

echo "Running the docker image [GPUS: ${GPU_ID}]"
docker_image="ysheng147/3d_object_labeling:v0.5"

# Start docker image
docker  run -d -it\
$VOLUMES \
$ENVS \
$VISUAL \
--mount type=bind,source=$CODE_PATH,target=/MS3D \
--gpus "device="$GPU_ID \
--privileged \
--net=host \
--ipc=host \
--shm-size=60G \
--workdir=/MS3D \
$docker_image   

#--runtime=nvidia \
