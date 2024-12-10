#!/bin/bash

docker run --runtime=nvidia --name court-iq-container --ipc host --env-file .env --gpus all --rm -p 8888:8888 -it -v ./:/root/ court-iq