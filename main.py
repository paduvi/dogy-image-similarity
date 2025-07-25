from flask import Flask, request, jsonify
from waitress import serve
import argparse

app = Flask(__name__)


@app.route('/api/echo', methods=['POST'])
def echo():
    data = request.json
    return jsonify({"you_sent": data}), 200


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RESTful API Server')
    parser.add_argument('--port', type=int, help='Server Port', required=True)
    args = parser.parse_args()

    print(f"🚀 Running server via waitress at http://localhost:{args.port}")
    serve(app, host='0.0.0.0', port=args.port)
