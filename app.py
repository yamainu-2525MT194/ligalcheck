import os
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import PyPDF2
import openai
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from ml_trainer import ContractLegalAnalyzer

load_dotenv()

ml_analyzer = ContractLegalAnalyzer()

# Global variables for ML training
CURRENT_MODEL = None
CURRENT_MODEL_TYPE = None
TRAINING_DATA = None
TRAINING_LABELS = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

openai.api_key = os.getenv('OPENAI_API_KEY')

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def analyze_contract(contract_text):
    try:
        # First, use ML model for initial risk assessment
        ml_risk_level = ml_analyzer.predict_risk(contract_text)
        
        # Then use OpenAI for detailed analysis
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal contract analysis assistant. Provide a detailed risk assessment and key observations."},
                {"role": "user", "content": f"Analyze this contract for potential risks and key legal considerations. Initial ML risk assessment is: {ml_risk_level}\n\n{contract_text}"}
            ]
        )
        detailed_analysis = response.choices[0].message.content
        
        return f"機械学習リスクレベル: {ml_risk_level}\n\n詳細分析:\n{detailed_analysis}"
    except Exception as e:
        return f"Error in analysis: {str(e)}"

@app.route('/upload', methods=['POST'])
def upload_contract():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        try:
            contract_text = extract_text_from_pdf(filepath)
            analysis_result = analyze_contract(contract_text)
            
            return jsonify({
                "filename": filename,
                "analysis": analysis_result
            }), 200
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/ml_trainer', methods=['GET'])
def ml_trainer():
    return render_template('ml_trainer.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    global TRAINING_DATA, TRAINING_LABELS
    
    if 'datasets' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    files = request.files.getlist('datasets')
    
    try:
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join('datasets/raw_contracts', filename)
            file.save(file_path)
        
        # Assuming CSV with 'text' and 'label' columns
        df = pd.read_csv(file_path)
        TRAINING_DATA = df['text'].values
        TRAINING_LABELS = df['label'].values
        
        return jsonify({'success': True, 'message': 'データのアップロードに成功'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

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
    app.run(debug=True, port=5000)
