# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, redirect, render_template
from inference import get_prediction

app = Flask(__name__)
@app.route("/predict", methods=['GET', 'POST'])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        #file=request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)

    # return jsonify({'class_id': class_id, "class_name": class_name})
        return render_template('result.html', class_id=class_id, class_name=class_name)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

