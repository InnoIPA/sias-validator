#!/bin/bash

FILE=$(realpath "$0")
ROOT=$(dirname $(dirname "${FILE}"))
cd $ROOT
source docker/config.sh

docker build \
-f docker/Dockerfile \
-t ${NAME}:${VER} .