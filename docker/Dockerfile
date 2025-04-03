FROM ubuntu:latest
    MAINTAINER  <juraph@rovaan>


# Add juraph (me!)
RUN useradd -ms /bin/bash juraph
WORKDIR /home/juraph

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get clean

RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    lsb-release \
    vim \
    ranger \
    feh \
    fdclone \
    ripgrep \
    zsh \
    python3-pip \
    && apt-get clean


USER juraph
RUN python3 -m pip config set global.break-system-packages true
RUN pip install cvxpy==1.6.1 cycler==0.12.1 flax==0.10.4 matplotlib==3.10.0 numpy==2.2.2 optax==0.2.4 pandas==2.2.3 scipy==1.15.1 setuptools==59.6.0

ENV TERM xterm-256color
