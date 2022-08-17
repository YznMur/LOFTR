#!/bin/bash
docker exec -it loftr\
    /bin/bash -c "
    cd /home/trainer/loftr;
    nvidia-smi;
    /bin/bash"