import os

import torch
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("cuDNN enabled:", torch.backends.cudnn.enabled)
    print("cuDNN version:", torch.backends.cudnn.version())

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

for file in client.files.list():
    print(file.name)
    client.files.delete(name=file.name)

print("All files deleted.")
