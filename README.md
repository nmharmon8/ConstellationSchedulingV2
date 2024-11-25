# Getting Started 🚀

## Setting up the nvidia docker containers 🐳

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


## Starting Docker Containers For ConstSched (Backend and UI) 🖥️

```bash
cd docker
docker compose up backend ui
```

If you're running on a remote server you will need to forward the ports to your local machine.
```bash
ssh -L 3000:localhost:3000 -L 4000:localhost:4000 username@server_address
```

## Starting TaskGPT 🤖
Task GPT is not required but the taskGPT command bar will not work without it. Task GPT must be on the same machine as the backend.
```bash
cd TaskGPT
docker compose up ollama
docker exec -it ollama /bin/bash
ollama pull llama3.2:3b
ollama pull all-minilm
exit
docker compose down
docker compose up
```

You can now view the UI at http://localhost:3000 and use the UI.



## Training your own model 🧠

Bring up the constsched docker container.
```bash
cd docker
docker compose up constsched -d
```

Attach your terminal to the container.
```bash
docker exec -it <container_id> /bin/bash
```

```bash
python rl/train.py --config=rl/configs/train_config.yaml --name=<name>
```

You will likely want to adjust the parameters in the train_config.yaml file to better suit your training needs.

train_config.yaml is located in the rl/configs folder.

```bash
# If you want to resume training from a checkpoint
resume: true
checkpoint_path: "/data/nm/v187_geo/PPO_2024-11-25_17-03-30/"  # Set this to the checkpoint path when resuming

# Set you log directory
log_dir: "/data/nm"

# You must set the resources appropriately for your environment. If you don't have the number of CPUs you claim it will not run.
env_runners:
  sample_timeout_s: 1000000.0
  num_env_runners: 24
  num_cpus_per_env_runner: 1
```

For logs start tensorboard in your log directory.
```bash
tensorboard --logdir <log_dir>
```


Have fun with ConstSched and RL! 🎮
