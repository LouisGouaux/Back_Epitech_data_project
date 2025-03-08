from flask import Flask, jsonify, request, Response
import json
from main import get_resources, get_flags_by_year

app = Flask(__name__)


@app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({
        'message': 'PONG'
    })

@app.route('/api/get-daily_data')
def get_daily_data():
    date = request.args.get('date')
    df = get_resources(date)
    response = Response(json.dumps(df, ensure_ascii=False), content_type="application/json; charset=utf-8")
    return response

@app.route('/api/calendar')
def get_calendar():
    year = request.args.get('year')
    df = get_flags_by_year(year)
    response = Response(json.dumps(df, ensure_ascii=False), content_type="application/json; charset=utf-8")
    return response


if __name__ == '__main__':
    app.run(debug=True)
