from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from utils import full_pca_process

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['PLOT_FOLDER'] = 'static/plots'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filename = secure_filename(image.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(img_path)

            data = full_pca_process(img_path, app.config['OUTPUT_FOLDER'], app.config['PLOT_FOLDER'])

            return render_template('index.html', **data)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
