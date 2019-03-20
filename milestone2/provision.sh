#!/bin/bash

apt-get update && apt-get upgrade
apt-get install -y git bzip2

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh


git clone https://github.com/tbonza/CS7180.git
