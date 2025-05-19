import os
import json
import logging
import logging.handlers
import hashlib
from datetime import datetime, timedelta, timezone
import pandas as pd
import pypdf
from flask import (
    Flask, request, jsonify, send_from_directory, session,
    redirect, url_for, flash, abort, make_response, current_app
)
from flask_caching import Cache
from flask_session import Session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException, BadRequest, Unauthorized, Forbidden, NotFound, InternalServerError
import jwt
import pdfplumber
import joblib
from asgiref.wsgi import WsgiToAsgi
import asyncio
from dotenv import load_dotenv
from japanese_contract_analyzer import JapaneseContractAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import uuid
import asyncio
import secrets
from functools import wraps, partial

load_dotenv()

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException, BadRequest, Unauthorized, Forbidden, NotFound, InternalServerError
import secrets
import traceback
import sys
from functools import wraps

# Session configuration constants
SESSION_TYPE = 'filesystem'
SESSION_FILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flask_sessions')
SESSION_COOKIE_NAME = 'legal_review_session'
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
SESSION_COOKIE_SAMESITE = 'Lax'
PERMANENT_SESSION_LIFETIME = timedelta(days=7)
SESSION_USE_SIGNER = True

# アプリケーションの初期化
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB 制限
    SESSION_TYPE = SESSION_TYPE
    SESSION_FILE_DIR = SESSION_FILE_DIR
    SESSION_COOKIE_NAME = SESSION_COOKIE_NAME
    SESSION_COOKIE_SECURE = SESSION_COOKIE_SECURE
    SESSION_COOKIE_HTTPONLY = SESSION_COOKIE_HTTPONLY
    SESSION_COOKIE_SAMESITE = SESSION_COOKIE_SAMESITE
    PERMANENT_SESSION_LIFETIME = PERMANENT_SESSION_LIFETIME
    SESSION_USE_SIGNER = SESSION_USE_SIGNER
    SESSION_COOKIE_PATH = '/'
    SESSION_COOKIE_DOMAIN = None
    SESSION_REFRESH_EACH_REQUEST = True
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5000']
    CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    CORS_HEADERS = ['Content-Type', 'Authorization']
    CACHE_TYPE = 'FileSystemCache'
    CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
    CACHE_DEFAULT_TIMEOUT = 300  # 5分
    CACHE_THRESHOLD = 1000  # キャッシュの最大アイテム数

app = Flask(__name__)
app.config.from_object(Config())

from flask import render_template

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# ASGIアプリケーションを作成
asgi_app = WsgiToAsgi(app)

# 必要なディレクトリが存在することを確認
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# 必要なディレクトリが存在することを確認
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(SESSION_FILE_DIR, exist_ok=True)

# セッションの初期化
Session(app)

# セッションクッキー名を明示的に設定
app.config['SESSION_COOKIE_NAME'] = SESSION_COOKIE_NAME
app.config['SESSION_COOKIE_HTTPONLY'] = SESSION_COOKIE_HTTPONLY
app.config['SESSION_COOKIE_SECURE'] = SESSION_COOKIE_SECURE
app.config['SESSION_COOKIE_SAMESITE'] = SESSION_COOKIE_SAMESITE
app.config['PERMANENT_SESSION_LIFETIME'] = PERMANENT_SESSION_LIFETIME

# CORS設定
CORS(app, 
     resources={r"/*": {"origins": app.config['CORS_ORIGINS']}},
     supports_credentials=True,
     expose_headers=['Content-Type', 'X-CSRFToken'],
     allow_headers=['Content-Type', 'Authorization', 'X-CSRFToken'])

# キャッシュの設定
app.config['CACHE_TYPE'] = 'FileSystemCache'
app.config['CACHE_DIR'] = os.path.join(os.path.dirname(__file__), 'cache')
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5分
app.config['CACHE_THRESHOLD'] = 1000  # キャッシュの最大アイテム数

# キャッシュの初期化
cache = Cache(app)

# キャッシュディレクトリが存在することを確認
os.makedirs(app.config['CACHE_DIR'], exist_ok=True)

# ログ設定
# ログディレクトリの作成
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

# ログローテーション設定
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 標準ログハンドラー（デバッグ）
debug_handler = logging.handlers.RotatingFileHandler(
    os.path.join(log_dir, 'debug.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(log_formatter)

# エラーログハンドラー
error_handler = logging.handlers.RotatingFileHandler(
    os.path.join(log_dir, 'error.log'),
    maxBytes=5*1024*1024,  # 5MB
    backupCount=3
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(log_formatter)

# アプリケーションロガーの設定
logger = logging.getLogger('legal_review_app')
logger.setLevel(logging.DEBUG)
logger.addHandler(debug_handler)
logger.addHandler(error_handler)

# Flaskアプリケーションのロガー設定
app.logger.addHandler(debug_handler)
app.logger.addHandler(error_handler)
app.logger.setLevel(logging.DEBUG)

# ファイルアップロードのセキュリティチェック
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'csv', 'xlsx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(file):
    max_size = app.config['MAX_CONTENT_LENGTH']
    if file.content_length > max_size:
        raise ValueError(f'File size exceeds maximum allowed size ({max_size/1024/1024}MB)')

def secure_filename(filename):
    """安全なファイル名を生成する
    
    Args:
        filename (str): 元のファイル名
        
    Returns:
        str: 安全なファイル名
    """
    # ファイル名の長さ制限
    if len(filename) > 255:
        filename = filename[:255]
    return filename

# APIキーの検証
def verify_api_key():
    api_key = request.headers.get('X-API-Key')
    if not api_key or api_key != os.getenv('API_KEY'):
        abort(401, 'Invalid API key')

# 認証デコレータ
def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            abort(403, 'Authentication required')
        return f(*args, **kwargs)
    return decorated_function

# CORSミドルウェア
def cors_middleware():
    if request.method == 'OPTIONS':
        return '', 204
    return None

# カスタム例外クラス
class APIError(Exception):
    """APIエラー用のカスタム例外クラス"""
    def __init__(self, message, status_code=400, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or {})
        rv['status'] = 'error'
        rv['message'] = self.message
        return rv

# エラーハンドラ
@app.errorhandler(APIError)
def handle_api_error(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'status': 'error',
        'message': 'リソースが見つかりません'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"サーバーエラー: {str(error)}")
    app.logger.error(traceback.format_exc())
    return jsonify({
        'status': 'error',
        'message': '内部サーバーエラーが発生しました'
    }), 500

@app.errorhandler(Exception)
def handle_exception(error):
    # HTTP例外の場合はそのまま返す
    if isinstance(error, HTTPException):
        return error
    
    # その他の例外はログに記録して500エラーを返す
    app.logger.error(f"未処理の例外: {str(error)}")
    app.logger.error(traceback.format_exc())
    
    return jsonify({
        'status': 'error',
        'message': '予期せぬエラーが発生しました',
        'error': str(error)
    }), 500

# リクエスト前後の処理
@app.before_request
def before_request():
    """リクエスト前の処理"""
    app.logger.info(f"Request: {request.method} {request.path}")
    app.logger.debug(f"Headers: {dict(request.headers)}")
    if request.method in ['POST', 'PUT']:
        app.logger.debug(f"Request data: {request.get_json(silent=True) or request.form.to_dict()}")

@app.after_request
def after_request(response):
    """レスポンス前の処理"""
    app.logger.info(f"Response: {response.status_code}")
    # エラーレスポンスのロギング
    if 400 <= response.status_code < 600:
        app.logger.error(f"Error response: {response.get_data(as_text=True)}")
    return response

def async_route(f):
    """非同期関数をFlaskのルートで使用できるようにするデコレータ"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        # 新しいイベントループを作成
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 非同期関数を実行
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            # イベントループを閉じる
            if not loop.is_closed():
                loop.close()
    return wrapper

# エラーハンドリング付きのデコレータ
def handle_errors(f):
    """エラーハンドリングを行うデコレータ"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(f):
                loop = asyncio.get_event_loop()
                try:
                    return loop.run_until_complete(f(*args, **kwargs))
                except RuntimeError as e:
                    if "no running event loop" in str(e):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
            else:
                return f(*args, **kwargs)
        except APIError as e:
            return jsonify({
                'status': 'error',
                'message': e.message,
                'code': e.status_code
            }), e.status_code
        except Exception as e:
            app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': 'An unexpected error occurred',
                'code': 500
            }), 500
    return decorated_function

app.before_request(cors_middleware)

# Global variables for ML training
CURRENT_MODEL = None
CURRENT_MODEL_TYPE = None
TRAINING_DATA = None
TRAINING_LABELS = None

def extract_text_from_pdf(file_path):
    """PDFファイルからテキストを抽出する（pypdf使用）"""
    try:
        from pypdf import PdfReader
        text = ''
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
        return text.strip()
    except Exception as e:
        app.logger.error(f'Error extracting text from PDF: {str(e)}')
        app.logger.error(traceback.format_exc())
        raise

def analyze_contract(pdf_file):
    # 日本語契約書分析モジュールを使用
    analyzer = JapaneseContractAnalyzer()
    
    # pdfplumberでテキスト抽出
    import pdfplumber
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    
    # 不要な空白を削除
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        analysis_result = analyzer.analyze_contract(text)
        return {
            'risk_level': analysis_result['risk_level'],
            'risk_score': round(analysis_result['risk_score'], 2),
            'explanation': analysis_result['explanation']
        }
    except Exception as e:
        return {
            'error': f"分析中にエラーが発生: {str(e)}"
        }

def handle_contract_upload(request):
    import traceback
    import logging
    import sys
    import os
    
    # Ensure log directory exists
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(log_dir, 'contract_upload_debug.log'), 
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.debug('=' * 50)
    logging.debug('Starting contract upload process')
    logging.debug(f'Request method: {request.method}')
    logging.debug(f'Request content type: {request.content_type}')
    logging.debug(f'Request files: {list(request.files.keys())}')
    logging.debug(f'Request form data: {dict(request.form)}')
    
    # Log all files in the request
    for key, file in request.files.items():
        logging.debug(f'File key: {key}')
        logging.debug(f'Filename: {file.filename}')
        logging.debug(f'Content type: {file.content_type}')
    
    debug_info = {}
    try:
        data = request.get_data()
        debug_info['headers'] = dict(request.headers)
        debug_info['form'] = request.form.to_dict(flat=False)
        debug_info['files'] = list(request.files.keys())
        # ログ出力
        with open('request_debug.log', 'a', encoding='utf-8') as f:
            f.write('==== REQUEST HEADERS ====\n')
            f.write(str(debug_info['headers']) + '\n')
            f.write('==== REQUEST FORM ====\n')
            f.write(str(debug_info['form']) + '\n')
            f.write('==== REQUEST FILES ====\n')
            f.write(str(debug_info['files']) + '\n')
            f.write('==== REQUEST DATA (first 200 bytes) ====\n')
            f.write(str(data[:200]) + '\n\n')
        # ファイル保存
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        from datetime import datetime
        import uuid
        safe_filename = f"contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
        if 'file' in request.files:
            file = request.files['file']
        elif 'contract' in request.files:
            file = request.files['contract']
        else:
            debug_info['error'] = 'ファイルが見つかりません'
            return jsonify({'error': 'ファイルが見つかりません', 'debug': debug_info}), 400
        # ファイル名をASCII安全名に変換し、PDFsディレクトリに直接保存
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pdfs_dir = os.path.abspath(os.path.join(base_dir, 'datasets/raw_contracts/pdfs'))
        ann_dir = os.path.abspath(os.path.join(base_dir, 'datasets/raw_contracts/annotations'))
        os.makedirs(pdfs_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        # 安全なファイル名生成
        import unicodedata
        import re
        def safe_filename(filename):
            # 日本語文字を含むファイル名をASCII安全名に変換
            filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('utf-8')
            filename = re.sub(r'[^\w\s-]', '', filename).strip()
            return filename

        # タイムスタンプ付きの安全なファイル名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = secure_filename(file.filename)
        safe_base_name = safe_filename(os.path.splitext(original_filename)[0])
        pdf_filename = f"{safe_base_name}_{timestamp}.pdf"
        ann_filename = f"{safe_base_name}_{timestamp}.txt"

        # PDFを直接保存先に保存
        logging.debug(f'Preparing to save PDF to {pdfs_dir}')
        logging.debug(f'PDF filename: {pdf_filename}')
        logging.debug(f'File object details: filename={file.filename}, content_type={file.content_type}')
        logging.debug(f'PDFs directory exists: {os.path.exists(pdfs_dir)}')
        
        # ディレクトリ作成を確認
        os.makedirs(pdfs_dir, exist_ok=True)
        os.makedirs(upload_dir, exist_ok=True)
        
        # PDF保存
        pdf_dst = os.path.join(pdfs_dir, pdf_filename)
        try:
            file.save(pdf_dst)
            logging.debug(f'PDF saved successfully to {pdf_dst}')
            logging.debug(f'PDF file exists after save: {os.path.exists(pdf_dst)}')
            logging.debug(f'PDF file size: {os.path.getsize(pdf_dst)} bytes')
        except Exception as e:
            logging.error(f'Error saving PDF: {e}')
            logging.error(f'Traceback: {traceback.format_exc()}')
            raise

        # アップロードディレクトリにもコピー
        upload_file_path = os.path.join(upload_dir, safe_filename)
        try:
            shutil.copy(pdf_dst, upload_file_path)
            logging.debug(f'PDF copied to upload directory: {upload_file_path}')
        except Exception as e:
            logging.error(f'Error copying PDF to upload directory: {e}')
            logging.error(f'Traceback: {traceback.format_exc()}')

        # デバッグ情報記録
        debug_info['saved_file_path'] = pdf_dst
        debug_info['upload_file_path'] = upload_file_path
        debug_info['pdf_filename'] = pdf_filename

        # PDF分析
        analysis_result = analyze_contract(pdf_dst)
        debug_info['analysis_result'] = analysis_result

        # データセット保存テスト
        try:
            dataset_path = 'datasets/raw_contracts/sample_contracts.csv'
            import pandas as pd
            import os
            # 必要なフィールド取得
            risk_level = request.form.get('risk_level', '')
            legal_review = request.form.get('legal_review', '')
            risk_factors = request.form.get('risk_factors', '[]')
            additional_notes = request.form.get('notes', '')
            debug_info['dataset_fields'] = {
                'risk_level': risk_level,
                'legal_review': legal_review,
                'risk_factors': risk_factors,
                'additional_notes': additional_notes,
                'text': text
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(dataset_path, index=False)
            debug_info['csv_saved'] = True

            # アノテーション保存
            ann_dst = os.path.join(ann_dir, ann_filename)
            with open(ann_dst, 'w', encoding='utf-8') as f:
                f.write(f"リスクレベル: {risk_level}\n")
                f.write(f"リーガルチェック: {legal_review}\n")
                f.write(f"リスク要因: {risk_factors}\n")
                f.write(f"追加ノート: {additional_notes}\n")
                f.write(f"テキスト冒頭: {text[:200]}\n")
            debug_info['annotation_saved_to'] = ann_dst
            debug_info['annotation_save_success'] = True
            debug_info['annotation_exists_after_save'] = os.path.exists(ann_dst)
            if not os.path.exists(ann_dst):
                debug_info['annotation_save_error'] = f"ファイルが生成されませんでした: {ann_dst}"
        except Exception as e:
            debug_info['csv_error'] = traceback.format_exc()
        if 'error' in analysis_result:
            return jsonify({'error': analysis_result['error'], 'debug': debug_info}), 500
        return jsonify({
            "filename": filename,
            "analysis": f"リスクレベル: {analysis_result.get('risk_level', '')}\n"
                       f"リスクスコア: {analysis_result.get('risk_score', 0):.2f}\n\n"
                       f"分析結果:\n{analysis_result.get('explanation', '')}",
            "debug": debug_info
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_details = traceback.format_exc()
        app.logger.error(f'Unexpected upload error: {str(e)}', exc_info=True)
        debug_info['exception'] = error_details
        return jsonify({'error': 'アップロード中にエラーが発生しました', 'debug': debug_info}), 500

@app.route('/upload', methods=['POST'])
@handle_errors
@async_route
async def upload_contract():
    print("[DEBUG] Start upload_contract")
    app.logger.info("Entered upload_contract") # Add this line
    """非同期で契約書をアップロードして分析するエンドポイント"""
    if 'file' not in request.files:
        print("[DEBUG] No file in request.files")
        raise APIError("ファイルがアップロードされていません", 400)
    
    file = request.files['file']
    print(f"[DEBUG] Uploaded filename: {file.filename}")
    if not file or file.filename == '':
        print("[DEBUG] No filename or file is empty")
        raise APIError("ファイルが選択されていません", 400)
    
    if not allowed_file(file.filename):
        raise APIError("許可されていないファイル形式です。PDFファイルをアップロードしてください。", 400)
    
    try:
        # ファイルのバリデーション
        validate_file_size(file)
        
        # 一時ファイルとして保存
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        
        # キャッシュキーの生成（ファイルのハッシュ値を使用）
        file_hash = hashlib.md5(open(filepath, 'rb').read()).hexdigest()
        cache_key = f"contract_analysis_{file_hash}"
        
        # キャッシュから結果を取得、または分析を実行
        result = cache.get(cache_key)
        if result is None:
            # テキスト抽出
            text = extract_text_from_pdf(filepath)
            if not text.strip():
                raise APIError("PDFからテキストを抽出できませんでした。別のファイルをお試しください。", 400)
            
            # 非同期で分析を実行
            analyzer = JapaneseContractAnalyzer()
            # 非同期関数を実行
            result = await analyzer.analyze_contract_async(text)
            
            # 結果をシリアライズ可能な形式に変換
            serializable_result = {
                'risk_level': result.get('risk_level'),
                'risk_score': float(result.get('risk_score', 0.0)),  # numpy.float32 を float に変換
                'explanation': result.get('explanation', '')
            }
            
            # 結果をキャッシュ（5分間有効）
            cache.set(cache_key, serializable_result, timeout=300)
            result = serializable_result
        
        return jsonify({
            'status': 'success',
            'message': '契約書の分析が完了しました',
            'data': result
        })
        
    except Exception as e:
        app.logger.error(f"契約書の分析中にエラーが発生しました: {str(e)}")
        app.logger.error(traceback.format_exc())
        raise APIError("契約書の分析中にエラーが発生しました", 500)

@app.route('/ml_trainer', methods=['GET', 'POST'])
@handle_errors
def ml_trainer():
    # 現在の学習データ数を取得
    dataset_path = 'datasets/raw_contracts/sample_contracts.csv'
    try:
        df = pd.read_csv(dataset_path)
        dataset_count = len(df)
    except FileNotFoundError:
        dataset_count = 0
    if request.method == 'POST':
        if 'file' not in request.files:
            raise APIError("ファイルがアップロードされていません", 400)
        
        file = request.files['file']
        if not file or file.filename == '':
            raise APIError("ファイルが選択されていません", 400)
        
        if not allowed_file(file.filename):
            raise APIError("許可されていないファイル形式です。CSVまたはXLSXファイルをアップロードしてください。", 400)
        
        try:
            validate_file_size(file)
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # ファイルの検証（必要に応じて実装）
            # validate_training_file(filepath)
            
            return jsonify({
                'status': 'success',
                'message': 'トレーニングファイルが正常にアップロードされました',
                'filename': filename
            })
        except ValueError as e:
            raise APIError(str(e), 400)
        except Exception as e:
            app.logger.error(f"トレーニングファイルの処理中にエラーが発生しました: {str(e)}")
            app.logger.error(traceback.format_exc())
            raise APIError("トレーニングファイルの処理中にエラーが発生しました", 500)
    
    # GETリクエストの場合はトレーナーページを表示
    return render_template('ml_trainer.html', dataset_count=dataset_count)

@app.route('/get_dataset_count', methods=['GET'])
@handle_errors
def get_dataset_count():
    # 現在の学習データ数を取得
    dataset_path = 'datasets/raw_contracts/sample_contracts.csv'
    try:
        df = pd.read_csv(dataset_path)
        dataset_count = len(df)
    except FileNotFoundError:
        dataset_count = 0
    
    return jsonify({'count': dataset_count})

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        # データセット読み込み
        dataset_path = 'datasets/raw_contracts/sample_contracts.csv'
        df = pd.read_csv(dataset_path)
        
        if len(df) < 1:
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
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description='Legal Review App')
    parser.add_argument('--port', type=int, default=5002, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run the server in debug mode')
    parser.add_argument('--asgi', action='store_true', help='Run as ASGI server (uvicorn)')
    args = parser.parse_args()
    
    if args.asgi:
        # ASGIサーバーで実行
        uvicorn.run(
            'app:asgi_app',
            host='0.0.0.0',
            port=args.port,
            reload=args.debug,
            log_level='info'
        )
    else:
        # 通常のWSGIサーバーで実行
        app.run(host='0.0.0.0', port=args.port, debug=args.debug)
