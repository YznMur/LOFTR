#!/bin/bash
cd "$(dirname "$0")"
cd ..

workspace_dir=$PWD

if ["$(docker ps -aq -f status=exited -f name=loftr)" ]; then
    docker rm loftr
fi

docker run -it -d --rm \
    --gpus all  \
    --net host \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -e "DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    --shm-size="35g" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --name loftr \
    -v $workspace_dir/:/home/trainer/loftr:rw \
    -v /home/user/Datasets/:/home/trainer/Datasets/:rw \
    x64/loftr:latest


    # \ '"device=0"'
