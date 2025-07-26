import argparse
import os
import sys
import torch
from torchvision.models import resnet18
from flask import Flask, request, jsonify
from waitress import serve

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from demo_resnet import extract_sample_time
from util import cut_video

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

# --- Prepare model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extract_model = resnet18(weights='DEFAULT')
extract_model.fc = torch.nn.Identity()
extract_model.to(device)
extract_model.eval()

app = Flask(__name__)


@app.route('/api/echo', methods=['POST'])
def echo():
    data = request.json
    return jsonify({"you_sent": data}), 200


@app.route('/api/video/query', methods=['POST'])
def query():
    data = request.json
    chosen_time = extract_sample_time(data["input_path"], data["query_image"], extract_model)
    return str(chosen_time)


@app.route('/api/video/cut', methods=['POST'])
def cut():
    data = request.json
    cut_video(data["input_path"], data["output_path"], data["start_time"])
    return 'Success'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RESTful API Server')
    parser.add_argument('--port', type=int, help='Server Port', required=True)
    args = parser.parse_args()

    print(f"🚀 Running server via waitress at http://localhost:{args.port}")
    serve(app, host='0.0.0.0', port=args.port)
