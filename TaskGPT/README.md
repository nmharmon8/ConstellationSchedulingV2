## Quick Start


Start ollama and download the models
```
docker compose up ollama
docker exec -it ollama /bin/bash
ollama pull llama3.2:3b
ollama pull all-minilm
exit
docker compose down
```

Then start everything
```
docker compose up
```
Test the client
```
cd client
python3 dummy_client.py "image ports in china"
```
Shutdown Task GPT
```
docker compose down
```

Clean up volumes if needed (delete weavia data and ollama models)
```
docker volume rm weaviate_data
docker volume rm ollama_data
```

## Notes
Port 5983 is exposed by default and will accept POST messages to /api/generate with a "prompt" json parameter.

The initialization of taskgpt_server rebuilds the weaviate databases. This takes around 5-10 seconds, but it may take longer if they are expanded. Any active file changes will bounce the flask app and rebuild weaviate.




