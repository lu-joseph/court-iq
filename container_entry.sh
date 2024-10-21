#!/bin/bash
screen -d -m -S jupyter-screen jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
exec /bin/bash