import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import json
import PyPDF2

class ContractLegalAnalyzer:
    def __init__(self, raw_data_dir='datasets/raw_contracts', 
                 processed_data_dir='datasets/processed', 
                 model_dir='models'):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.model_dir = model_dir
        
    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text
    
    def prepare_training_data(self):
        """
        Prepare training data from PDFs and annotations
        Expects a JSON file with annotations for each PDF
        """
        contracts = []
        labels = []
        
        for filename in os.listdir(self.raw_data_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.raw_data_dir, filename)
                annotation_path = os.path.join(self.raw_data_dir, filename.replace('.pdf', '.json'))
                
                if os.path.exists(annotation_path):
                    text = self.extract_text_from_pdf(pdf_path)
                    with open(annotation_path, 'r') as f:
                        annotations = json.load(f)
                    
                    contracts.append(text)
                    labels.append(annotations.get('risk_level', 'unknown'))
        
        df = pd.DataFrame({
            'contract_text': contracts,
            'risk_level': labels
        })
        
        df.to_csv(os.path.join(self.processed_data_dir, 'contract_data.csv'), index=False)
        return df
    
    def train_model(self):
        df = pd.read_csv(os.path.join(self.processed_data_dir, 'contract_data.csv'))
        
        # Text vectorization
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['contract_text'])
        
        # Label encoding
        le = LabelEncoder()
        y = le.fit_transform(df['risk_level'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        # Save model and vectorizer
        joblib.dump(clf, os.path.join(self.model_dir, 'contract_risk_model.pkl'))
        joblib.dump(vectorizer, os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'))
        joblib.dump(le, os.path.join(self.model_dir, 'label_encoder.pkl'))
    
    def predict_risk(self, contract_text):
        """
        Predict risk level for a new contract
        """
        model = joblib.load(os.path.join(self.model_dir, 'contract_risk_model.pkl'))
        vectorizer = joblib.load(os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'))
        le = joblib.load(os.path.join(self.model_dir, 'label_encoder.pkl'))
        
        X = vectorizer.transform([contract_text])
        prediction = model.predict(X)
        return le.inverse_transform(prediction)[0]

def main():
    analyzer = ContractLegalAnalyzer()
    analyzer.prepare_training_data()
    analyzer.train_model()

if __name__ == '__main__':
    main()
