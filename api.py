from flask import Flask, jsonify, Response
import json
from main import get_resources

app = Flask(__name__)


@app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({
        'message': 'PONG'
    })

@app.route('/api/get-daily_data')
def get_daily_data():
    df = get_resources('2024-02-01')
    response = Response(json.dumps(df, ensure_ascii=False), content_type="application/json; charset=utf-8")
    return response


if __name__ == '__main__':
    app.run(debug=True)
