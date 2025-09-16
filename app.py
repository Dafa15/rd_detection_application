import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing import image
import tensorflow as tf

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class LayerScale(Layer):
    def __init__(self, init_values=1e-5, projection_dim=512, *args, **kwargs):
        self.init_values = init_values
        self.projection_dim = projection_dim
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.projection_dim,),
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({
            'init_values': self.init_values,
            'projection_dim': self.projection_dim,
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

def crop_image(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PREPROCESSED_FOLDER = 'static/preprocessed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREPROCESSED_FOLDER'] = PREPROCESSED_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)

model = load_model('model.keras', custom_objects={'LayerScale': LayerScale})
CLASS_NAMES = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def apply_clahe(img, clip_limit=2.0):
    if not CV2_AVAILABLE:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dalam permintaan.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        try:
            if CV2_AVAILABLE:
                img = cv2.imread(upload_path)
                if img is None:
                    return jsonify({'error': 'Gagal membaca gambar.'}), 400

                img = crop_image(img)
                processed_img = apply_clahe(img, clip_limit=2.0)

                processed_filename = 'processed_' + filename
                processed_path = os.path.join(app.config['PREPROCESSED_FOLDER'], processed_filename)
                cv2.imwrite(processed_path, processed_img)

                img_for_model = cv2.resize(processed_img, (224, 224))
                img_for_model = cv2.cvtColor(img_for_model, cv2.COLOR_BGR2RGB)

            else:
                img = image.load_img(upload_path, target_size=(224, 224))
                processed_path = None
                img_for_model = img

            img_array = image.img_to_array(img_for_model)
            img_array = np.expand_dims(img_array, 0)

            predictions = model.predict(img_array)
            predicted_index = tf.argmax(predictions[0]).numpy()
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = round(float(np.max(predictions[0])) * 100, 2)

            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence,
                'original_url': '/' + upload_path.replace('\\', '/'),
                'processed_url': '/' + processed_path.replace('\\', '/') if processed_path else ''
            })

        except Exception as e:
            return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {str(e)}'}), 500

    return jsonify({'error': 'Format file tidak didukung.'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
