#!/bin/bash

docker run --name court-iq-container --ipc host --gpus all --rm -p 8888:8888 -it -v ./:/root/ court-iq