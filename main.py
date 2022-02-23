from flask import Flask, request
from flask_restful import Api
from flask_cors import CORS

from src.interface import *

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)


@app.route("/inference", methods=["POST"])
def inference_by_time():
    """
    Input:
    - Dict(segment_id: List, time: List)
    - Dict(segment_id: List, time: int)

    Output:
    - Dict(segment_id: List, LOS: List)
    """
    req = request.json  # dictionary
    if type(req["time"]) is not list:
        req["time"] = [req["time"]] * len(req["asegment_id"])
    return time_inference(req)


@app.route("/period_inference", methods=["POST"])
def inference_by_period():
    """
    Input:
    - Dict(segment_id: List, date: str, period: List)
    - Dict(segment_id: List, date: str, period: str)

    Output:
    - Dict(segment_id: List, LOS: List)
    """
    req = request.json  # dictionary
    req["date"] = [req["date"]] * len(req["segment_id"])
    if type(req["period"]) is not list:
        req["period"] = [req["period"]] * len(req["segment_id"])
    return period_inference(req)


@app.route("/seq_inference", methods=["POST"])
def seq_inference():
    req = request.json
    return sequence_inference(req)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
