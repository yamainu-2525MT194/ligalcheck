import pytest
import os
import sys
import asyncio
import tempfile
import shutil
import pytest_asyncio
from pathlib import Path
from typing import Generator, Dict, Any, Optional, AsyncGenerator
from datetime import timedelta

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

# アプリケーションをインポート
from app import app

# テスト用の一時ディレクトリを設定
TEST_UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'legal_review_test_uploads')

# テスト用の設定
@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """テスト用の設定を返す"""
    # 一時ディレクトリを作成
    
    # テスト用のアップロードフォルダを作成
    os.makedirs(TEST_UPLOAD_FOLDER, exist_ok=True)
    
    # テスト用の一時ファイルを保存するディレクトリを設定
    app.config['UPLOAD_FOLDER'] = TEST_UPLOAD_FOLDER
    
    # テスト用のセッション設定
    app.config['SESSION_TYPE'] = 'null'  # テスト中はセッションを無効化
    app.config['TESTING'] = True
    
    # モデルとベクトルライザーのパスを設定
    model_dir = os.path.join(TEST_UPLOAD_FOLDER, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    config = {
        'MODEL_PATH': os.path.join(model_dir, 'test_model.joblib'),
        'VECTORIZER_PATH': os.path.join(model_dir, 'test_vectorizer.joblib'),
        'UPLOAD_FOLDER': TEST_UPLOAD_FOLDER
    }
    
    # テスト用のクライアントを提供
    with app.app_context():
        yield config
    
    # テスト後に一時ファイルを削除
    shutil.rmtree(TEST_UPLOAD_FOLDER, ignore_errors=True)

@pytest.fixture(scope="function")
def temp_dir():
    """テスト用の一時ディレクトリを提供"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # テスト後にディレクトリを削除
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def client(app):
    """テスト用のクライアントを提供"""
    return app.test_client()

@pytest.fixture
def runner(app):
    """CLIコマンドのテスト用ランナーを提供"""
    return app.test_cli_runner()

# イベントループフィクスチャ
@pytest.fixture
def event_loop():
    """各テスト用の新しいイベントループを提供"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
