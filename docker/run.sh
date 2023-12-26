#!/bin/bash
FILE=$(realpath "$0")
ROOT=$(dirname $(dirname "${FILE}"))
cd $ROOT
source docker/config.sh

ARGS=$1

MODE="-it"
if [[ $ARGS = 'b' ]];then
    MODE="-dt"
    echo "Background Mode";
fi

docker run ${MODE} --rm \
--gpus all \
-w /workspace \
-v $(pwd):/workspace \
--ipc=host \
${NAME}:${VER} bash