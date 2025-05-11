import os
import json
import logging
from datetime import datetime
import pandas as pd
import PyPDF2
from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import joblib
import numpy as np
from japanese_contract_analyzer import JapaneseContractAnalyzer
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

load_dotenv()

app = Flask(__name__)
CORS(app)

# ロギング設定
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for ML training
CURRENT_MODEL = None
CURRENT_MODEL_TYPE = None
TRAINING_DATA = None
TRAINING_LABELS = None

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def analyze_contract(pdf_file):
    # 日本語契約書分析モジュールを使用
    analyzer = JapaneseContractAnalyzer()
    
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    try:
        analysis_result = analyzer.analyze_contract(text)
        return f"""リスクレベル: {analysis_result['risk_level']}
リスクスコア: {analysis_result['risk_score']:.2f}

分析結果:
{analysis_result['explanation']}"""    
    except Exception as e:
        return f"分析中にエラーが発生: {str(e)}"
        return f"Error in analysis: {str(e)}"

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_data()
    # ログ出力
    with open('request_debug.log', 'a', encoding='utf-8') as f:
        f.write('==== REQUEST HEADERS ====\n')
        f.write(str(dict(request.headers)) + '\n')
        f.write('==== REQUEST DATA (first 200 bytes) ====\n')
        f.write(str(data[:200]) + '\n\n')
    # ファイル保存
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    from datetime import datetime
    import uuid
    safe_filename = f"contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
    pdf_path = os.path.join(upload_dir, safe_filename)
    with open(pdf_path, 'wb') as f:
        f.write(data)
    return jsonify({'success': True, 'message': f'File saved as {safe_filename}'})

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
    console_handler.setFormatter(console_formatter)
    app.logger.addHandler(console_handler)
    
    # ファイルロガー設定
    import os
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'upload_debug.log')
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    app.logger.addHandler(file_handler)
    
    # ログファイルパスをログ
    print(f'Log file created at: {log_file_path}')
    
    # ログレベル設定
    app.logger.setLevel(logging.DEBUG)
    
    try:
        # リクエスト情報の詳細ログ
        app.logger.debug('Upload route started')
        app.logger.debug(f'Request method: {request.method}')
        app.logger.debug(f'Request content type: {request.content_type}')
        app.logger.debug(f'Request headers: {dict(request.headers)}')
        app.logger.debug(f'Request form data: {dict(request.form)}')
        app.logger.debug(f'Request files keys: {list(request.files.keys())}')
        
        # ファイル名安全化
        from datetime import datetime
        import uuid
        def generate_safe_filename():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = uuid.uuid4().hex[:8]
            return f'contract_{timestamp}_{unique_id}.pdf'
        safe_filename = generate_safe_filename()
        # ファイルサイズの確認
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        app.logger.debug(f'File size: {file_size} bytes')
        
        if file_size == 0:
            app.logger.error('Empty file uploaded')
            return jsonify({"error": "Empty file"}), 400
        
        if file_size > 10 * 1024 * 1024:  # 10MB以上のファイルを拒却
            app.logger.error('File too large')
            return jsonify({"error": "File size exceeds 10MB limit"}), 400
        
        # アップロードディレクトリ作成
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        # ファイル保存
        pdf_path = os.path.join(upload_dir, safe_filename)
        with open(pdf_path, 'wb') as f:
            f.write(file.read())
        app.logger.debug(f'File saved to: {pdf_path}')
        
        # 元のファイル名を保存
        with open(os.path.join(upload_dir, 'original_filenames.log'), 'a') as f:
            f.write(f'{safe_filename}: {file.filename}\n')
        
        return jsonify({
            "success": True,
            "filename": safe_filename,
            "message": "File uploaded successfully"
        }), 200
    
    except Exception as e:
        # 予期しないエラーを詳細にログ
        app.logger.error(f'Unexpected error during upload: {str(e)}')
        app.logger.error(traceback.format_exc())
        return jsonify({
            "error": "Upload failed", 
            "details": str(e)
        }), 500
        
        # テキスト抽出
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()
            app.logger.info(f'Extracted text length: {len(text)} characters')
        except Exception as text_error:
            app.logger.error(f'Text extraction error: {str(text_error)}')
            return jsonify({"error": f"Text extraction failed: {str(text_error)}"}), 500
        
        # 契約書分析
        try:
            analyzer = JapaneseContractAnalyzer()
            analysis_result = analyzer.analyze_contract(text)
        except Exception as analysis_error:
            app.logger.error(f'Contract analysis error: {str(analysis_error)}')
            return jsonify({"error": f"Contract analysis failed: {str(analysis_error)}"}), 500
        
        return jsonify({
            "filename": filename,
            "analysis": f"リスクレベル: {analysis_result['risk_level']}\n"
                       f"リスクスコア: {analysis_result['risk_score']:.2f}\n\n"
                       f"分析結果:\n{analysis_result['explanation']}"
        }), 200
    
    except Exception as e:
        # 予期しないエラーのロギング
        app.logger.error(f'Unexpected upload error: {str(e)}', exc_info=True)
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/ml_trainer', methods=['GET'])
def ml_trainer():
    # 現在の学習データ数を取得
    dataset_path = 'datasets/raw_contracts/sample_contracts.csv'
    try:
        df = pd.read_csv(dataset_path)
        dataset_count = len(df)
    except FileNotFoundError:
        dataset_count = 0
    return render_template('ml_trainer.html', dataset_count=dataset_count)

@app.route('/get_dataset_count', methods=['GET'])
def get_dataset_count():
    # 現在の学習データ数を取得
    dataset_path = 'datasets/raw_contracts/sample_contracts.csv'
    try:
        df = pd.read_csv(dataset_path)
        dataset_count = len(df)
    except FileNotFoundError:
        dataset_count = 0
    
    return jsonify({'count': dataset_count})

    legal_review = request.form.get('legal_review', '')
    risk_factors = request.form.get('risk_factors', '[]')
    additional_notes = request.form.get('notes', '')
    
    # リスク要因をJSONからリストに変換
    try:
        risk_factors = json.loads(risk_factors)
    except json.JSONDecodeError:
        risk_factors = []
    
    if contract_file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if contract_file:
        # ファイル保存
        filename = secure_filename(contract_file.filename)
        pdf_path = os.path.join('datasets/raw_contracts/pdfs', filename)
        with open(pdf_path, 'wb') as f:
            f.write(contract_file.read())
        
        # テキスト抽出
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        # アノテーションファイル作成
        annotation_filename = os.path.splitext(filename)[0] + '.txt'
        annotation_path = os.path.join('datasets/raw_contracts/annotations', annotation_filename)
        with open(annotation_path, 'w', encoding='utf-8') as f:
            f.write(f'リスクレベル: {risk_level}\n')
            f.write(f'追加メモ: {additional_notes}\n')
            f.write(f'アップロード日時: {datetime.now()}\n')
        
        # CSVデータセットに追加
        dataset_path = 'datasets/raw_contracts/sample_contracts.csv'
        try:
            df = pd.read_csv(dataset_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df = pd.DataFrame(columns=['text', 'risk_level', 'legal_review', 'risk_factors', 'additional_notes'])
        
        # 新しい行を追加
        new_row = pd.DataFrame({
            'text': [text], 
            'risk_level': [risk_level],
            'legal_review': [legal_review],
            'risk_factors': [', '.join(risk_factors)],
            'additional_notes': [additional_notes]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        
        df.to_csv(dataset_path, index=False)
        
        return jsonify({
            'success': True, 
            'message': '契約書と学習データが正常に保存されました', 
            'dataset_count': len(df)
        })
    
    return jsonify({'success': False, 'message': 'アップロードに失敗しました'})

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # データセット読み込み
        dataset_path = 'datasets/raw_contracts/sample_contracts.csv'
        df = pd.read_csv(dataset_path)
        
        if len(df) < 10:
            return jsonify({
                'success': False, 
                'message': '学習に必要な最小データ数に達していません。少なくとも1件以上の契約書をアップロードしてください。'
            })
        
        # 既存のアナライザーを使用
        analyzer = JapaneseContractAnalyzer()
        
        # モデル再学習
        accuracy = analyzer.train_model()
        
        return jsonify({
            'success': True, 
            'message': f'モデル再学習が完了しました。精度: {accuracy * 100:.2f}%'
        })
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'モデル再学習中にエラーが発生しました: {str(e)}'
        })


@app.route('/configure_model', methods=['POST'])
def configure_model():
    global CURRENT_MODEL, CURRENT_MODEL_TYPE
    
    data = request.json
    model_type = data.get('model_type')
    max_depth = int(data.get('max_depth', 10))
    
    try:
        if model_type == 'random_forest':
            CURRENT_MODEL = RandomForestClassifier(max_depth=max_depth)
            CURRENT_MODEL_TYPE = 'ランダムフォレスト'
        elif model_type == 'svm':
            CURRENT_MODEL = SVC(kernel='rbf')
            CURRENT_MODEL_TYPE = 'サポートベクターマシン'
        elif model_type == 'neural_network':
            CURRENT_MODEL = MLPClassifier(max_iter=1000)
            CURRENT_MODEL_TYPE = 'ニューラルネットワーク'
        
        return jsonify({'success': True, 'model': CURRENT_MODEL_TYPE})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/start_training', methods=['POST'])
def start_training():
    global CURRENT_MODEL, TRAINING_DATA, TRAINING_LABELS
    
    if CURRENT_MODEL is None or TRAINING_DATA is None or TRAINING_LABELS is None:
        return jsonify({'success': False, 'message': 'モデルまたはデータが設定されていません'})
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(TRAINING_DATA, TRAINING_LABELS, test_size=0.2, random_state=42)
        
        CURRENT_MODEL.fit(X_train, y_train)
        y_pred = CURRENT_MODEL.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred).tolist()
        
        # Save model
        model_path = os.path.join('models', f'{CURRENT_MODEL_TYPE}_model.pkl')
        joblib.dump(CURRENT_MODEL, model_path)
        
        return jsonify({
            'success': True,
            'evaluation': {
                'classification_report': report,
                'confusion_matrix': confusion,
                'model_type': CURRENT_MODEL_TYPE
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
    
    parser = argparse.ArgumentParser(description='Legal Review App')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run the server in debug mode')
    args = parser.parse_args()
    app.run(debug=args.debug, port=args.port)
