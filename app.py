from flask import Flask, render_template, request
from model_utils import load_and_train_model, predict_future
import os, pandas as pd
import tempfile
import json
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

@app.route('/', methods=['GET', 'POST'])
def index():
    chart_data = False
    prices, vols, dates, avg_vol = [], [], [], None

    if request.method == 'POST':
        file = request.files['csv_file']
        start_date = request.form['start_date']
        num_days = int(request.form['num_days'])

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        model_data = load_and_train_model(filepath)
        future_dates, predictions = predict_future(model_data, start_date, num_days)

        prices = [p[0] for p in predictions]
        vols = [p[1] for p in predictions]
        dates = [d.strftime('%Y-%m-%d') for d in future_dates]
        avg_vol = round(sum(vols) / len(vols), 4)
        chart_data = True
        print(prices, vols, dates, avg_vol)


    return render_template('index.html',
                           chart_data=chart_data,
                           prices=json.dumps(prices),
                           vols=json.dumps(vols),
                           dates=json.dumps(dates),
                           avg_vol=avg_vol)


if __name__ == '__main__':
    app.run(debug=True)
