import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transaction_exploration import graphical_analysis

path = '/Users/lucariotto/Documents/Personal/Gestione denaro/Analisi spese/Gestione entrate-spese.xlsx'
transaction_df = pd.read_excel(path, sheet_name='Transazioni')

# page title
st.title("Transaction Analysis")

# plot the monthly expenses 
transact_plot = graphical_analysis(transaction_df)
all_montly_exp = transact_plot.montly_ewma_plot(column_name='IMPORTO', alpha=0.1, date_name = 'DATA')
st.pyplot(all_montly_exp)

# Monthly expenditure chart by category
transact_plot = graphical_analysis(transaction_df)
category = transaction_df['CATEGORIA'].unique().tolist()
choice = st.selectbox("Select the category for compute the monthly expenses", category)
st.write(f"Monthly expenditure chart for the {choice} category")
choice_montly_exp=transact_plot.montly_ewma_plot(column_name='IMPORTO', alpha=0.1, date_name = 'DATA', category=choice)
st.pyplot(choice_montly_exp)