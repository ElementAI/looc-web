import os

from inference import get_prediction
from flask import Flask, render_template, request, redirect


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if not file:
            raise Exception('File not found')
        img_bytes = file.read()
        count = get_prediction(image_bytes=img_bytes)
        return render_template('result.html', count=count)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
