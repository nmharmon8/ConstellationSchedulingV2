from datetime import datetime
import weaviate
import json
import ollama
import time
from flask import Flask, request, jsonify, Response, stream_with_context
import subprocess

app = Flask(__name__)

# Initialize Weaviate client and collections
while True:
    try:
        client = weaviate.connect_to_local()
        client.is_live()  # Check if Weaviate is live
        print("Connected to Weaviate successfully.")
        break
    except Exception as e:
        print("Waiting for Weaviate to be available...")
        time.sleep(5)  # Wait before retrying

print("Running build_api_database.py...")
subprocess.run(["python3", "./db/build_api_database.py"], check=True)
print("Running build_database.py...")
subprocess.run(["python3", "./db/build_database.py"], check=True)
print("Database setup completed successfully.")

client = weaviate.connect_to_local()
collection = client.collections.get(name="docs")
collection_apis = client.collections.get(name="apis")


def parse_json_from_text(text):
    try:
        start = text.find("```")
        start_bracket = text.find("[")

        if start != -1:
            end = text.find("```", start + 3)
            if end != -1:
                json_text = text[start + 3:end].strip()
            else:
                return "Error: Closing triple quotes not found."
        elif start_bracket > 0:
            end = text.find("]", start_bracket)
            if end != -1:
                json_text = text[start_bracket:end+1].strip()
            else:
                return "Error: Closing ] not found."
        else:
            json_text = text.strip()

        data = json.loads(json_text)
        if type(data) is not list:
            data = list(data)
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        raise e


def generate_tasks(prompt):
    response = ollama.embeddings(model="all-minilm", prompt=prompt)
    results = collection.query.near_vector(near_vector=response["embedding"], limit=20)
    context = ""
    for i, result in enumerate(results.objects):
        context += f"{i}: " + result.properties['text'] + '\n'

    yield json.dumps({"status": "info", "context": context}) + '\n'

    api_format = collection_apis.query.near_vector(near_vector=response["embedding"], limit=1)
    api_format_text = ""
    if len(api_format.objects) == 1:
        api_format_text = api_format.objects[0].properties['text'].replace("'", "\"")
        yield json.dumps({"status": "info", "api_format": api_format_text}) + '\n'

    prompt_template = f"Use only the relevant data from this context: {context}. Respond to this data collection request: {prompt} {api_format_text}. Do not provide any text other than the json message."

    output = ""
    t = time.time()
    max_time = 300

    for chunk in ollama.generate(model="llama3.2:3b", prompt=prompt_template, stream=True):
        output += chunk['response']
        if time.time() - t > max_time:
            break
        yield json.dumps({"status": "llm_output", "chunk": chunk['response']}) + '\n'

    parsed_data = parse_json_from_text(output)
    yield json.dumps({"status": "final_output", "data": parsed_data}) + '\n'


@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        return Response(stream_with_context(generate_tasks(prompt)), mimetype='application/json')
    except Exception as e:
        print(f"Exception in generate: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="localhost", port=5983, debug=True)
