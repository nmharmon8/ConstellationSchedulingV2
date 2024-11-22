FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Update and install necessary tools
RUN apt-get update || true

# Helpful for Debian systems -  all packages need to compile such as gcc and g++ compilers and other utils.
RUN apt-get install -y build-essential python3 python3-setuptools python3-dev python3-tk python3-pip swig


RUN pip install ollama
RUN pip install -U weaviate-client
RUN apt install -y x11-apps
RUN pip install plotly pandas tqdm dash dash-dangerously-set-inner-html pillow numpy  

