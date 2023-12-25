import os
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

app = Flask(__name__)

# アップロードする画像を保存するディレクトリを設定
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 画像の保存先ディレクトリが存在しない場合、作成
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# サポートするファイルの拡張子を設定
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# ファイルの拡張子をチェックするための関数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 画像を読み込み、前処理するための関数
def load_and_preprocess_image(file_path, target_size=(150, 150)):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("指定したファイルは存在しません。")
    try:
        img = Image.open(file_path)
        img = img.convert("RGB")  
        img = img.resize(target_size)  
        return img_to_array(img)
    except Exception as e:
        print(f"エラー詳細: {str(e)}")
        raise ValueError("画像の読み込みおよび前処理中にエラーが発生しました。") from e

# 画像のアップロードと分類を処理するルート
@app.route('/', methods=['GET', 'POST'])
def upload_and_classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'ファイルが選択されていません。'
        file = request.files['file']
        if file.filename == '':
            return 'ファイル名が空です。'
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # 機械学習モデルを読み込み、画像を分類
            try:
                model_path = os.path.abspath("C:/CompSche/PythonRun/app.py/models/fashion_model.h5")
                model = load_model(model_path)
                image_data = load_and_preprocess_image(file_path)
                prediction = model.predict(np.expand_dims(image_data, axis=0))
                label = 'Yes' if prediction[0][0] > 0.5 else 'No'
                
                if label == 'Yes':
                    return redirect(url_for('oshare_html', uploaded_image_filename=file.filename))
                else:
                    return redirect(url_for('dasai_html', uploaded_image_filename=file.filename))

            except Exception as e:
                return str(e)
        else:
            return 'サポートされていないファイルの拡張子です。サポートされている拡張子は次のとおりです: {}。'.format(', '.join(ALLOWED_EXTENSIONS))
    return render_template('index.html')

# 'oshare.html' ページへのルート設定
@app.route('/oshare.html')
def oshare_html():
    uploaded_image_filename = request.args.get('uploaded_image_filename', default='', type=str)
    return render_template('oshare.html', uploaded_image_filename=uploaded_image_filename)

# 'dasai.html' ページへのルート設定
@app.route('/dasai.html')
def dasai_html():
    uploaded_image_filename = request.args.get('uploaded_image_filename', default='', type=str)
    label = request.args.get('label', default='', type=str)
    return render_template('dasai.html', uploaded_image_filename=uploaded_image_filename)

#ローカルで立ち上げる際のメイン部分(下記のコメントアウトを外して、172.16.0.180のメインをコメントアウトしてください)
# if __name__ == '__main__':
#     app.run(debug=True)

# アプリケーションサーバーで立ち上げる際のメイン部分
if __name__ == '__main__':
    app.run(debug=False, host='192.168.1.108', port=50003)