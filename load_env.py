import os
from dotenv import load_dotenv

def ensure_env_loaded():
    load_dotenv()
    assert os.environ.get('OPENAI_API_KEY'), "OPENAI_API_KEY is not set!"
