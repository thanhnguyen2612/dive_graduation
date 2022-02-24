from flask import Flask, request
from flask_restful import Api
from flask_cors import CORS
import sys

from src.interface import *

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)


@app.route("/inference", methods=["POST"])
def normal_inference():
    """
    @params
    - Dict(segment_ids: List, timestamp: int | List)
    @return
    - Dict(segment_ids: List, LOSes: List)
    """
    req = request.json  # dictionary
    try:
        return inference(req)
    except Exception as e:
        print(e, file=sys.stderr)
        return 'Failed'

@app.route("/seq_inference", methods=["POST"])
def seq_inference():
    """
    @params:
    - Dict(segment_ids: List, timestamp: int)
    @return:
    - Dict(segment_ids: List, LOSes: List)
    """
    req = request.json
    try:
        return sequence_inference(req)
    except Exception as e:
        print(e, file=sys.stderr)
        return 'Failed'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
