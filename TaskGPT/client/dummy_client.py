import requests
import sys
import json

def send_prompt(prompt):
    url = "http://localhost:5983/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt}

    try:
        response = requests.post(url, json=data, headers=headers, stream=True)

        if response.status_code == 200:
            # Buffer to collect chunks for final JSON output
            output_data = ""
            final_output_data = None

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    data = json.loads(line)
                    status = data.get("status")

                    # Display information based on status type
                    if status == "info":
                        if "context" in data:
                            print("\n--- Context Retrieved ---")
                            print(data["context"])
                        elif "api_format" in data:
                            print("\n--- API Format Retrieved ---")
                            print(data["api_format"])

                    elif status == "llm_output":
                        chunk = data.get("chunk", "")
                        print(chunk, end="", flush=True)  # Stream each chunk without extra line breaks
                        output_data += chunk

                    elif status == "final_output":
                        final_output_data = data.get("data")
            
            # Print the final parsed output nicely
            if final_output_data:
                print("\n\n--- Final Parsed JSON Output ---")
                print(json.dumps(final_output_data, indent=2))

        else:
            print(f"Error: Received status code {response.status_code}")
            print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <prompt>")
        sys.exit(1)

    prompt = sys.argv[1]
    send_prompt(prompt)
