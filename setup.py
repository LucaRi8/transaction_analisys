"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

from setuptools import setup

APP = ['run_streamlit.py']
DATA_FILES = ['transaction_exploration.py', '/Users/lucariotto/Documents/Personal/Gestione denaro/Analisi spese/Gestione entrate-spese.xlsx']
OPTIONS = {}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
