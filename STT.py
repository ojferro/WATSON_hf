import requests

API_TOKEN = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-base"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("sample1.flac")