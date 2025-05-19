import os
import re
import pandas as pd
import pdfplumber
import csv

PDF_DIR = 'datasets/raw_contracts/pdfs'
ANNOTATION_DIR = 'datasets/raw_contracts/annotations'
OUTPUT_CSV = 'datasets/raw_contracts/sample_contracts.csv'

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text.strip()

def extract_risk_level(annotation_path):
    with open(annotation_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(r'リスクレベル: (.+)', line)
            if m:
                return m.group(1).strip()
    return ''

def main():
    data = []
    for pdf_name in os.listdir(PDF_DIR):
        if not pdf_name.lower().endswith('.pdf'):
            continue
        base = os.path.splitext(pdf_name)[0]
        annotation_candidates = [base + '.txt', base.replace('.docx','') + '.txt']
        annotation_path = None
        for candidate in annotation_candidates:
            path = os.path.join(ANNOTATION_DIR, candidate)
            if os.path.exists(path):
                annotation_path = path
                break
        if not annotation_path:
            continue
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        text = extract_text_from_pdf(pdf_path)
        risk_level = extract_risk_level(annotation_path)
        if text and risk_level:
            data.append({'text': text, 'risk_level': risk_level})
    if data:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
        print(f'Generated {OUTPUT_CSV} with {len(df)} rows.')
    else:
        print('No data found.')

if __name__ == '__main__':
    main()
