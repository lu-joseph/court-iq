#!/bin/bash
docker image rm court-iq
sudo rm -rf .jupyter
sudo rm -rf .local
sudo rm -rf .ipython
docker build -t court-iq .