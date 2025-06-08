import os
import glob
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
import hashlib
import re

# Define paths
# Assuming this script is in the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEIYAKUSHO_KEI_DIR = "/Users/yamamotoyuuki/Documents/windsurf_dev/legal_review_app/契約書系"
DATASETS_RAW_DIR = "/Users/yamamotoyuuki/Documents/windsurf_dev/legal_review_app/datasets/raw_contracts"
ANNOTATIONS_DIR = os.path.join(DATASETS_RAW_DIR, "annotations")
PDFS_DIR = os.path.join(DATASETS_RAW_DIR, "pdfs")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "compiled_training_data.csv")

# --- Text Extraction Functions ---
def extract_text_from_docx(filepath):
    try:
        doc = Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from DOCX {filepath}: {e}")
        return ""

def extract_text_from_pdf(filepath):
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {filepath}: {e}")
        return ""

def read_text_from_txt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text from TXT {filepath}: {e}")
        return ""

def extract_text_from_doc(filepath):
    print(f"Warning: .doc file {filepath} found. Text extraction for legacy .doc format is not implemented in this script.")
    return ""

def get_text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# --- Main Data Collection Logic ---
def collect_and_compile_data():
    print("DEBUG: Entered collect_and_compile_data() function.")
    collected_data = []
    processed_content_hashes = set() # To store (hash(contract_text), hash(memo_text))

    # Helper to add data if not duplicate
    def add_data_pair(contract_text, memo_text, contract_path, memo_path):
        if not contract_text or not memo_text:
            print(f"Skipping pair due to empty content. Contract: {contract_path}, Memo: {memo_path}")
            return
        
        contract_hash = get_text_hash(contract_text)
        memo_hash = get_text_hash(memo_text)
        
        if (contract_hash, memo_hash) in processed_content_hashes:
            print(f"Skipping duplicate content pair. Contract: {contract_path}, Memo: {memo_path}")
            return
        
        collected_data.append({
            'contract_path': contract_path,
            'memo_path': memo_path,
            'contract_text': contract_text,
            'memo_text': memo_text
        })
        processed_content_hashes.add((contract_hash, memo_hash))
        print(f"Added pair: Contract: {contract_path}, Memo: {memo_path}")

    # 1. Process 契約書系 directory
    print(f"Processing {KEIYAKUSHO_KEI_DIR}...")
    for root, _, files in os.walk(KEIYAKUSHO_KEI_DIR):
        contract_files = []
        memo_files = []
        for f_name in files:
            f_path = os.path.join(root, f_name)
            if f_name.lower().endswith(('.docx', '.pdf')):
                contract_files.append(f_path)
            elif f_name.lower().endswith('.txt') and ('リーガル' in f_name or 'リーガル' in f_name): #リーガル is with NFD normalization
                memo_files.append(f_path)
        
        if contract_files and memo_files:
            # Simple pairing: assumes one contract and one relevant memo per folder, or tries to match suffixes
            # This might need refinement if multiple contracts/memos exist in one subfolder
            for c_path in contract_files:
                c_base = os.path.splitext(os.path.basename(c_path))[0]
                # Try to find a memo with a similar base or suffix (e.g., _L, _N)
                found_memo_for_contract = False
                for m_path in memo_files:
                    m_base = os.path.splitext(os.path.basename(m_path))[0]
                    # Basic suffix match (e.g., "契約書_L" and "リーガル_L")
                    c_match = re.search(r'_([A-Z0-9]+)$', c_base, re.IGNORECASE)
                    m_match = re.search(r'_([A-Z0-9]+)$', m_base, re.IGNORECASE)
                    if (c_match and m_match and c_match.group(1) == m_match.group(1)) or len(memo_files) == 1:
                        contract_text = ""
                        if c_path.lower().endswith('.docx'):
                            contract_text = extract_text_from_docx(c_path)
                        elif c_path.lower().endswith('.pdf'):
                            contract_text = extract_text_from_pdf(c_path)
                        
                        memo_text = read_text_from_txt(m_path)
                        add_data_pair(contract_text, memo_text, c_path, m_path)
                        found_memo_for_contract = True
                        break # Paired this contract
                if not found_memo_for_contract and len(contract_files) == 1 and len(memo_files) == 1:
                    # Fallback if only one contract and one memo, and no suffix match worked
                    c_path_single = contract_files[0]
                    m_path_single = memo_files[0]
                    contract_text = ""
                    if c_path_single.lower().endswith('.docx'):
                        contract_text = extract_text_from_docx(c_path_single)
                    elif c_path_single.lower().endswith('.pdf'):
                        contract_text = extract_text_from_pdf(c_path_single)
                    memo_text = read_text_from_txt(m_path_single)
                    add_data_pair(contract_text, memo_text, c_path_single, m_path_single)

    # 2. Process datasets/raw_contracts (annotations and pdfs)
    print(f"Processing {ANNOTATIONS_DIR} and {PDFS_DIR}...")
    memo_files_annotations = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.txt"))
    
    all_contract_files_in_pdfs_dir = glob.glob(os.path.join(PDFS_DIR, "*.*"))
    contract_file_map = {os.path.basename(f): f for f in all_contract_files_in_pdfs_dir}

    for m_path in memo_files_annotations:
        memo_filename = os.path.basename(m_path)
        contract_filepath = None

        # Pattern 1: e.g., "A.docx.txt" (memo) -> "A.docx.pdf" (contract)
        match1 = re.match(r'^(.*?\.docx)\.txt$', memo_filename, re.IGNORECASE)
        if match1:
            potential_contract_name = match1.group(1) + ".pdf" # Assuming it becomes .pdf
            if potential_contract_name in contract_file_map:
                contract_filepath = contract_file_map[potential_contract_name]
            else: # Try original extension if .pdf not found
                 potential_contract_name_orig_ext = match1.group(1)
                 if potential_contract_name_orig_ext in contract_file_map:
                    contract_filepath = contract_file_map[potential_contract_name_orig_ext]

        # Pattern 2: e.g., "B.txt" (memo) -> "B.pdf" (contract)
        if not contract_filepath:
            match2 = re.match(r'^(.*?)\.txt$', memo_filename, re.IGNORECASE)
            if match2:
                base_name = match2.group(1)
                for ext in ['.pdf', '.docx', '.doc']:
                    potential_contract_name = base_name + ext
                    if potential_contract_name in contract_file_map:
                        contract_filepath = contract_file_map[potential_contract_name]
                        break
        
        # Pattern 3: e.g., "リーガル_I.txt" (memo) -> "業務委託基本契約案_I.docx" (contract)
        if not contract_filepath:
            legal_match = re.match(r'^(?:リーガル|リーガル)_([A-Z0-9]+)\.txt$', memo_filename, re.IGNORECASE)
            if legal_match:
                identifier = legal_match.group(1)
                for c_name_map, c_path_map in contract_file_map.items():
                    if f'_{identifier}' in os.path.splitext(c_name_map)[0]:
                        contract_filepath = c_path_map
                        break
        
        if contract_filepath:
            contract_text = ""
            ext_lower = contract_filepath.lower()
            if ext_lower.endswith('.docx'):
                contract_text = extract_text_from_docx(contract_filepath)
            elif ext_lower.endswith('.pdf'):
                contract_text = extract_text_from_pdf(contract_filepath)
            elif ext_lower.endswith('.doc'):
                contract_text = extract_text_from_doc(contract_filepath) # Will print warning
            
            memo_text = read_text_from_txt(m_path)
            add_data_pair(contract_text, memo_text, contract_filepath, m_path)
        else:
            print(f"Could not find matching contract for memo: {m_path}")

    # Create DataFrame and save
    if not collected_data:
        print("No data pairs were collected. CSV file will not be created.")
        return

    df = pd.DataFrame(collected_data)
    try:
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"Successfully collected {len(df)} data pairs.")
        print(f"Data saved to: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

if __name__ == "__main__":
    print("DEBUG: Script execution started from __main__.")
    collect_and_compile_data()
