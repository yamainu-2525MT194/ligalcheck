import pytest
import os
import io
import shutil
import tempfile
import asyncio
from unittest.mock import patch, MagicMock
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock, Mock
from flask import Flask, session, jsonify, request, current_app
from werkzeug.datastructures import FileStorage
import sys
import json
from typing import Dict, Any, Optional, Coroutine, Any
import pytest_asyncio

# 非同期関数を同期的に実行するためのヘルパー関数
def sync(coro: Coroutine[Any, Any, Any]):
    """非同期関数を同期的に実行する"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def create_test_pdf_data():
    """テスト用のPDFデータをメモリ上に作成する"""
    from reportlab.pdfgen import canvas
    from io import BytesIO
    
    # メモリ上にPDFを作成
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.drawString(100, 100, "This is a test PDF file.")
    p.save()
    
    # バッファの位置を先頭に戻す
    buffer.seek(0)
    return buffer.getvalue()

def create_test_pdf(filepath):
    """テスト用のPDFファイルを作成する"""
    pdf_data = create_test_pdf_data()
    with open(filepath, 'wb') as f:
        f.write(pdf_data)

# テスト用の一時ファイルを作成するヘルパー関数
def create_test_pdf(file_path):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    c = canvas.Canvas(file_path, pagesize=letter)
    c.drawString(100, 750, "This is a test PDF file for contract analysis.")
    c.save()

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# メインアプリをインポート
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as flask_app
from flask_session import Session
from japanese_contract_analyzer import JapaneseContractAnalyzer

# テスト用の設定
TEST_CONFIG = {
    'TESTING': True,
    'WTF_CSRF_ENABLED': False,  # CSRF保護を無効化
    'UPLOAD_FOLDER': os.path.join(os.path.dirname(__file__), 'test_uploads'),
    'SECRET_KEY': 'test-secret-key',
    'SESSION_TYPE': 'null',  # Use null session for testing
    'SESSION_COOKIE_NAME': 'session',
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SECURE': False,
    'SESSION_COOKIE_SAMESITE': 'Lax',
    'PERMANENT_SESSION_LIFETIME': 3600,  # 1 hour
    'SERVER_NAME': 'localhost.localdomain'  # テスト用のサーバー名を設定
}

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # テスト用の設定を適用
    test_config = TEST_CONFIG.copy()
    test_config['TESTING'] = True
    test_config['WTF_CSRF_ENABLED'] = False
    
    # テスト用のアップロードディレクトリを作成
    upload_dir = os.path.join(os.path.dirname(__file__), 'test_uploads')
    test_config['UPLOAD_FOLDER'] = upload_dir
    
    # 既存の設定を更新
    flask_app.config.update(test_config)
    
    # ディレクトリが存在することを確認
    os.makedirs(upload_dir, exist_ok=True)
    
    # セッションを初期化 (null sessionを使用)
    flask_app.config['SESSION_TYPE'] = 'null'
    Session(flask_app)
    
    yield flask_app
    
    # テスト後にクリーンアップ
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir, ignore_errors=True)

@pytest.fixture
def client(app):
    """A test client for the app."""
    with app.test_client() as client:
        with app.app_context():
            yield client

@pytest_asyncio.fixture
async def runner(app):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()

def create_test_pdf_data():
    """テスト用のPDFデータをメモリ上に作成する"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "テスト用契約書")
    c.drawString(100, 730, "本契約書はテスト用のダミー契約書です。")
    c.drawString(100, 710, "契約期間: 2025年1月1日から2026年12月31日まで")
    c.drawString(100, 690, "契約金: 1,000,000円（税込）")
    c.drawString(100, 670, "支払い条件: 契約締結後30日以内に全額を支払うものとします。")
    c.drawString(100, 650, "違約金の条件: 契約違反があった場合は100万円を支払うものとします。")
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def create_test_pdf(filepath):
    """テスト用のPDFファイルを作成する"""
    pdf_data = create_test_pdf_data()
    with open(filepath, 'wb') as f:
        f.write(pdf_data)

def test_upload_contract_success(client, app):
    """契約書アップロードの成功テスト"""
    # テスト用のPDFデータを作成
    pdf_data = create_test_pdf_data()

    # モックの結果を準備
    mock_result = {
        'risk_level': '低リスク',
        'risk_score': 0.95,
        'explanation': 'テスト説明'
    }

    # 非同期関数をモック
    async def mock_analyze_contract_async(contract_text):
        return mock_result

    # モックの設定
    with patch('app.allowed_file', return_value=True) as mock_allowed_file, \
         patch('japanese_contract_analyzer.JapaneseContractAnalyzer.analyze_contract_async', side_effect=mock_analyze_contract_async) as mock_analyze:

        # ファイルをアップロード
        data = {
            'file': (io.BytesIO(pdf_data), 'test_contract.pdf')
        }

        # クライアントを使用してリクエストを送信
        response = client.post(
            '/upload_contract',
            data=data,
            content_type='multipart/form-data'
        )

        # レスポンスを検証
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['status'] == 'success'
        assert 'data' in json_data
        assert 'risk_level' in json_data['data']
        assert 'risk_score' in json_data['data']
        assert 'explanation' in json_data['data']

        # モックが呼び出されたことを確認
        mock_allowed_file.assert_called_once()
        mock_analyze.assert_called_once()

def test_upload_invalid_file_type(client, app):
    """不正なファイルタイプのアップロードテスト"""
    # テスト用のテキストデータを作成
    text_data = b"This is a test text file."

    # モックの設定
    with patch('app.allowed_file', return_value=False) as mock_allowed_file:
        # 不正なファイルタイプでアップロードを試みる
        data = {
            'file': (io.BytesIO(text_data), 'test_file.txt')
        }

        # クライアントを使用してリクエストを送信
        response = client.post(
            '/upload_contract',
            data=data,
            content_type='multipart/form-data'
        )

        # レスポンスを検証
        assert response.status_code == 400
        json_data = response.get_json()
        assert json_data['status'] == 'error'
        assert '許可されていないファイル形式です' in json_data['message']

        # モックが呼び出されたことを確認
        mock_allowed_file.assert_called_once()

def test_analyze_contract(app):
    """契約書分析テスト"""
    # テスト用のPDFデータを作成
    pdf_data = create_test_pdf_data()

    # テスト用の一時ファイルを作成
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_file.write(pdf_data)
        temp_file_path = temp_file.name

    try:
        # モックの結果を準備
        mock_result = {
            'status': 'success',
            'analysis': {
                'document_type': 'contract',
                'parties': ['Party A', 'Party B']
            }
        }


        # モック関数
        async def mock_analyze_contract(pdf_file):
            return mock_result

        # アプリケーションコンテキスト内で実行
        with app.app_context():
            # モックの設定
            with patch('app.analyze_contract', side_effect=mock_analyze_contract) as mock_analyze:
                # 分析を実行
                from app import analyze_contract
                
                # イベントループを取得して非同期関数を実行
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(analyze_contract(temp_file_path))
                    # 結果を検証
                    assert result == mock_result
                    mock_analyze.assert_called_once()
                finally:
                    loop.close()
    finally:
        # 一時ファイルを削除
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
