

# Setting up the nvidia docker containers

Install the nvidia-docker2 package to support GPU containers.
```bash
sudo apt-get install -y nvidia-docker2
```

Add the Nvidia runtime configuration to /etc/docker/daemon.json
```
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```

Restart the docker service to pick up the new configuration.
```bash
sudo systemctl restart docker
```



# Vizard
```bash
Vizard_Linux/Vizard.x86_64 --args -loadFile /home/nharmon/git/NM/_VizFiles/vis_UnityViz.bin
```

# Tensorboard 
```bash
python -m tensorboard.main --logdir=.
```



# NN
Initilizing the layers in the NN for RAY with torch dose not happend by defualt. 