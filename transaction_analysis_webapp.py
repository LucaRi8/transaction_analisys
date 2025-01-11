import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transaction_exploration import graphical_analysis
import plotly.graph_objects as go
import plotly.express as px

path = '/Users/lucariotto/Documents/Personal/Gestione denaro/Analisi spese/Gestione entrate-spese.xlsx'
transaction_df = pd.read_excel(path, sheet_name='Transazioni')

# cleaning of data
cols_to_keep = [
    "DATA",
    "GIORNO",
    "MESE",
    "ANNO",
    "TIPO TRANSAZIONE",
    "CONTO",
    "CATEGORIA",
    "SOTTOCATEGORIA",
    "IMPORTO"
]
category_df = transaction_df.loc[transaction_df['CATEGORIE'].notna(), 'CATEGORIE']
transaction_df = transaction_df[cols_to_keep]
transaction_df.loc[:,'mese_anno'] = transaction_df['DATA'].dt.to_period('M')
mese_anno_df = pd.DataFrame(transaction_df['mese_anno'].unique(), columns=['mese_anno'])
df_outer_join = pd.merge(
    mese_anno_df,  # seleziona solo la colonna 'mese_anno'
    category_df,  # seleziona solo la colonna 'categoria'
    how='cross',  # tipo di join (outer per includere tutte le combinazioni)
    on=None,  # nessuna colonna su cui fare l'unione
)
df_outer_join.rename(columns={'CATEGORIE': 'CATEGORIA'}, inplace=True)

transaction_df = pd.merge(
    df_outer_join,
    transaction_df,
    how='left',
    on=['mese_anno', 'CATEGORIA']
)

transaction_df['IMPORTO'] = np.where(transaction_df['IMPORTO'].isna(), 0, transaction_df['IMPORTO'])

# fill some na column
exit_category = [
    "Casa",
    "Spesa",
    "Viaggi/Esperienze",
    "Trasporti",
    "Serate",
    "Pasti fuori",
    "Sport",
    "Abbonamenti",
    "Shopping",
    "Altro"
]
transaction_df['TIPO TRANSAZIONE'] = np.where(transaction_df['CATEGORIA'].isin(exit_category), 'Uscita', 'Entrata')

# inizialize data in graphical_analysis class 
transact_plot = graphical_analysis(transaction_df)

# page configuration
st.set_page_config(
    page_title="Transaction Analysis",  # Title of broswer page
    page_icon=":bar_chart :",  
    layout="wide"  # Layout 
)
page = st.sidebar.radio('Pages', options=['Expenses', 'Income', 'Assets'])

# window of mooving average selection
window = st.sidebar.number_input(
        "Insert the window size for mooving average for all expenses plot",
        min_value=2,  # min value
        value=10,      # default value
        step=1        # step
        )

if page == 'Expenses':
    # title of the page
    st.title("Analysis of expenses") 
    col1, col2 = st.columns(2)
    with col1:
        st.write('<p style="font-size:24px; color:white;">Monthly expenses time series</p>', unsafe_allow_html=True)
        st.write('Chart showing the aggregated monthly expenses and the moving average')
        # plot the monthly expenses 
        all_montly_exp = transact_plot.montly_ma_plot(column_name='IMPORTO', window=window, date_name = 'DATA')
        st.plotly_chart(all_montly_exp)

    with col2:
        st.write('<p style="font-size:24px; color:whitw;">Monthly cashflow time series</p>', unsafe_allow_html=True)
        st.write('Chart showing the aggregated monthly cashflof and the moving average')
        # plot the monthly expenses 
        # plot the monthly expenses 
        cash_flow_plot = transact_plot.cash_flow_plot(column_name='IMPORTO', window=window, date_name = 'DATA')
        st.plotly_chart(cash_flow_plot)

    # expeses for category analysis
    col1, col2 = st.columns(2)
    with col1:
        # table for montly expenses divided by category
        category_expenses = (transaction_df[transaction_df['TIPO TRANSAZIONE'] == 'Uscita']
                        .groupby(['mese_anno', 'CATEGORIA'])['IMPORTO']
                        .sum().reset_index())
        category_expenses = (category_expenses.groupby('CATEGORIA')['IMPORTO']
                             .rolling(window=window, min_periods=1).mean()
                             .reset_index().groupby('CATEGORIA')
                            .last().drop(['level_1'], axis=1))
        st.dataframe(category_expenses)  
    
    with col2:
        category_expenses.reset_index(inplace=True)
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=category_expenses['CATEGORIA'],        # Labels (categories)
                    values=category_expenses['IMPORTO'],           # Values
                    hoverinfo='label+percent+value', # Show label, percentage, and value on hover
                    textinfo='percent',           # Display only percentage on the pie chart
                    #marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']),  # Custom colors
                    )
                ]
        )
        # Update layout to include a legend
        fig.update_layout(
            title="Category Distribution",
            legend=dict(title="Categories", orientation="v", x=1, y=1),
        )
        st.plotly_chart(fig)


    # expeses for category analysis graphical visualization
    col1, col2 = st.columns(2)
    with col1:
        category = transaction_df.loc[transaction_df['TIPO TRANSAZIONE'] == 'Uscita', 'CATEGORIA'].unique().tolist()
        choice1 = st.selectbox("Select the category for compute the monthly expenses", category,  key="category_selectbox")
        if choice1 in exit_category:
            type_transaction = 'Uscita'
        else : 
            type_transaction = 'Entrata'
        st.write(f'<p style="font-size:24px; color:white;">Monthly expenditure chart for the {choice1} category</p>', 
                unsafe_allow_html=True)
        # plot the monthly expenses for category
        choice1_montly_exp=transact_plot.montly_ma_plot(column_name='IMPORTO', window=10, date_name = 'DATA', 
                                                        category=choice1, type_transaction=type_transaction)
        st.plotly_chart(choice1_montly_exp)

    with col2:
        subcategory = transaction_df.loc[transaction_df['TIPO TRANSAZIONE'] == 'Uscita', 'SOTTOCATEGORIA'].unique().tolist()
        subcategory = list(filter(pd.notna, subcategory))
        choice2 = st.selectbox("Select the subcategory for compute the monthly expenses", subcategory,  key="subcategory_selectbox")
        st.write(f'<p style="font-size:24px; color:white;">Monthly expenditure chart for the {choice2} subcategory</p>', 
                unsafe_allow_html=True)
        # plot the monthly expenses for category
        choice2_montly_exp=transact_plot.montly_ma_plot(column_name='IMPORTO', window=10, date_name = 'DATA', sub_category=choice2)
        st.plotly_chart(choice2_montly_exp)