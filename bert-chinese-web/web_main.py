from flask import Flask
from flask import render_template, request
from predict import Bert_summary_model
from config import iphost, port

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api_summary', methods=("GET", "POST"))
def api_summary():
    if request.method == "POST":
        info = request.values.to_dict()
        doc = info['doc']
        doc = doc.replace('\n', '')
        if len(doc) > sum_model.max_process_len:
            summary = sum_model.long_predict(doc)
        else:
            summary = sum_model.predict(doc)
        return summary
    else:
        return ""


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    sum_model = Bert_summary_model()
    app.run(host=iphost, port=port, debug=True)
    # app.run(host='127.0.0.1', port=8080, debug=True)
