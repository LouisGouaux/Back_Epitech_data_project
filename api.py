import pandas as pd
from flask import Flask, jsonify, request, Response
import json
from main import import_dataset, get_resources, get_flags_by_year, get_flags_by_month
from predict import create_futur_data, create_futur_monthly_calendar


app = Flask(__name__)


@app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({
        'message': 'PONG'
    })


@app.route('/api/get-daily_data')
def get_daily_data():
    request_date = request.args.get('date')
    df = import_dataset()
    df["date"] = pd.to_datetime(df["date"])
    df_last_date = df["date"].max()
    converted_date = pd.to_datetime(request_date)
    if (converted_date<df_last_date):
        df_result = get_resources(request_date)
    else:
        df_result = create_futur_data(request_date)

    response = Response(json.dumps(df_result, ensure_ascii=False), content_type="application/json; charset=utf-8")
    return response


@app.route('/api/calendar')
def get_calendar():
    year = request.args.get('year')
    if (request.args.get('filter-by') == 'month'):
        month = request.args.get('month')
        request_date = f"{year}-{month}-01"
        df = import_dataset()
        df["date"] = pd.to_datetime(df["date"])
        df_last_date = df["date"].max()
        converted_date = pd.to_datetime(request_date)
        if (converted_date < df_last_date):
            df = get_flags_by_month(year, month)
        else:
            df = create_futur_monthly_calendar(year, month)
    else:
        df = get_flags_by_year(year)
    response = Response(json.dumps(df, ensure_ascii=False), content_type="application/json; charset=utf-8")
    return response


if __name__ == '__main__':
    app.run(debug=True)
