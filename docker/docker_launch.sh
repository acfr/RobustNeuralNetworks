#!/usr/bin/env sh


docker container stop rnn_docker
docker container rm rnn_docker
docker run --network=host --ipc=host --pid=host --privileged --ulimit nofile=1024:2056 --env="DISPLAY" -it -v /dev/shm:/dev/shm -v ~/local_progs/RobustNeuralNetworks/:/home/RNN/rnn_ws:Z --name rnn_docker rnn_docker
