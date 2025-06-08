import os
import csv
import glob
from pathlib import Path

import docx
import pdfplumber

# 対象ディレクトリ
BASE_DIR = Path(__file__).parent.parent / '契約書系'
OUTPUT_CSV = Path(__file__).parent.parent / 'datasets/processed/contract_pairs.csv'

# ファイルパターン
patterns = [
    '**/*契約書*.pdf',
    '**/*契約書*.docx',
    '**/*契約書*.doc',
]

# テキスト抽出関数
def extract_text_from_pdf(filepath):
    try:
        with pdfplumber.open(filepath) as pdf:
            return '\n'.join([page.extract_text() or '' for page in pdf.pages])
    except Exception as e:
        print(f"[ERROR] PDF抽出失敗: {filepath}: {e}")
        return ''

def extract_text_from_docx(filepath):
    try:
        doc = docx.Document(filepath)
        return '\n'.join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"[ERROR] DOCX抽出失敗: {filepath}: {e}")
        return ''

def extract_text_from_doc(filepath):
    # doc形式はpython-docxで直接読めないため、LibreOfficeなどで変換推奨
    # ここでは未対応とする
    print(f"[WARN] DOC形式は未対応: {filepath}")
    return ''

# 既存CSVの読み込み（重複防止）
existing_texts = set()
if OUTPUT_CSV.exists():
    with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_texts.add(row['contract_text'].strip())

new_rows = []
for pattern in patterns:
    for filepath in glob.glob(str(BASE_DIR / pattern), recursive=True):
        filepath = Path(filepath)
        if filepath.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(filepath)
        elif filepath.suffix.lower() == '.docx':
            text = extract_text_from_docx(filepath)
        elif filepath.suffix.lower() == '.doc':
            text = extract_text_from_doc(filepath)
        else:
            continue
        text = text.strip()
        if not text or text in existing_texts:
            continue
        # 仮のリーガルチェック結果（要手動修正）
        legal_result = '要確認：この契約書の内容をレビューしてください。'
        new_rows.append({'contract_text': text, 'legal_check_result': legal_result})
        existing_texts.add(text)

# 追記
if new_rows:
    write_header = not OUTPUT_CSV.exists() or os.stat(OUTPUT_CSV).st_size == 0
    with open(OUTPUT_CSV, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['contract_text', 'legal_check_result'])
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)
    print(f"{len(new_rows)}件の契約書テキストを追記しました。")
else:
    print("新規追加データはありません。")
