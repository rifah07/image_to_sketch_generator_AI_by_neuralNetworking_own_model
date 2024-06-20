from flask import Flask, request, render_template, url_for
import os
from werkzeug.utils import secure_filename
from model import convert_to_sketch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    sketch_url = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            sketch_filepath = convert_to_sketch(filepath)
            sketch_url = url_for('static', filename=os.path.basename(sketch_filepath))
    return render_template('index.html', sketch_url=sketch_url)

if __name__ == "__main__":
    app.run(debug=True)
