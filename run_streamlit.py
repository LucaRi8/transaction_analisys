import subprocess
import sys

def run_streamlit():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "/Users/lucariotto/Documents/Personal/Gestione denaro/Analisi spese/transaction_analisys/transaction_analysis_webapp.py"])

if __name__ == "__main__":
    run_streamlit()
