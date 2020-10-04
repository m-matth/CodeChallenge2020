# CodeChallenge2020


# setup

/etc/apt.sourcing.list :

deb https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/ /


apt-get update
apt-get install libcudart10.1 libcudart10.2 libnvidia-tesla-450-cuda1 nvidia-cuda-dev:amd64

https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.140.tar.gz

tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-1.14.0.tar.gz

export LD_LIBRARY_PATH=/usr/local/lib

stack build

