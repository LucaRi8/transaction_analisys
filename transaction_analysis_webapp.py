import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transaction_exploration import graphical_analysis
import plotly.graph_objects as go
import plotly.express as px

path = '/Users/lucariotto/Documents/Personal/Gestione denaro/Analisi spese/Gestione entrate-spese.xlsx'
transaction_df = pd.read_excel(path, sheet_name='Transazioni')

# associations between catecories and subcategories
categories_associations = {
    "Casa": ["Affitto", "Bollette", "Pulizia", np.nan],
    "Spesa": [np.nan],
    "Viaggi/Esperienze": [np.nan],
    "Trasporti": ["Benzina", "Mezzi pubblici", "Mezzi a noleggio", "Autostrada", np.nan],
    "Serate": ["Alcool", "Entrate/Biglietti", np.nan],
    "Pasti fuori": ["Cene", "Pranzi", "Pranzi lavoro", np.nan],
    "Sport": ["Pass", "Attrezzatura", np.nan],
    "Abbonamenti": [np.nan],
    "Shopping": [np.nan],
    "Stipendio": [np.nan],
    "Altre entrate": [np.nan],
    "Giroconto entrata": [np.nan],
    "Giroconto uscita": [np.nan],
    "Altro": [np.nan]
}

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

# fill the missing categoriees and sub cateories with zero transaction value
subcat_df = pd.DataFrame(columns=['CATEGORIA', 'SOTTOCATEGORIA'])
for key in categories_associations:
    cat = [key] * len(categories_associations[key])
    subcat_df = pd.concat([subcat_df, pd.DataFrame({'CATEGORIA': cat, 
                                                   'SOTTOCATEGORIA' : categories_associations[key]})],
                                                     axis=0)
df_outer_join = pd.merge(
    mese_anno_df,  # seleziona solo la colonna 'mese_anno'
    category_df,  # seleziona solo la colonna 'categoria'
    how='cross',  # tipo di join (outer per includere tutte le combinazioni)
    on=None,  # nessuna colonna su cui fare l'unione
)
df_outer_join.rename(columns={'CATEGORIE': 'CATEGORIA'}, inplace=True)
df_outer_join = pd.merge(
    df_outer_join,
    subcat_df,
    on='CATEGORIA',
    how='left'
)
transaction_df = pd.merge(
    df_outer_join,
    transaction_df,
    how='left',
    on=['mese_anno', 'CATEGORIA', 'SOTTOCATEGORIA']
)
    
# fill some na column
transaction_df['IMPORTO'] = np.where(transaction_df['IMPORTO'].isna(), 0, transaction_df['IMPORTO'])
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
ciroconti_category = [
    'Giroconto uscita',
    'Giroconto entrata'
]
transaction_df['TIPO TRANSAZIONE'] = np.where(transaction_df['CATEGORIA'].isin(exit_category), 'Uscita', 
                                              np.where(transaction_df['CATEGORIA'].isin(ciroconti_category), 
                                                       'Giroconto', 'Entrata'))

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
        value=12,      # default value
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
        all_montly_exp = transact_plot.montly_ma_plot(column_name='IMPORTO', window=window, date_name = 'mese_anno')
        st.plotly_chart(all_montly_exp)

    with col2:
        st.write('<p style="font-size:24px; color:whitw;">Monthly cashflow time series</p>', unsafe_allow_html=True)
        st.write('Chart showing the aggregated monthly cashflof and the moving average')
        # plot the monthly expenses 
        cash_flow_plot = transact_plot.cash_flow_plot(column_name='IMPORTO', window=window, date_name = 'mese_anno')
        st.plotly_chart(cash_flow_plot)

    # expeses for category analysis
    col1, col2 = st.columns(2)
    year_month = (
        transaction_df
        .sort_values(by = 'mese_anno', ascending = False)['mese_anno']
        .astype('str')
        .unique()
        )
    year_month = list(year_month[:3]) + ['Mooving average']
    with col1:
        ym1 = st.selectbox("Select between the month of interest or a mooving average", year_month,  index=3)
        if ym1 == 'Mooving average':
            # table for montly expenses divided by category
            category_expenses = (transaction_df[transaction_df['TIPO TRANSAZIONE'] == 'Uscita']
                            .groupby(['mese_anno', 'CATEGORIA'])['IMPORTO']
                            .sum().reset_index())
            category_expenses = (category_expenses.groupby('CATEGORIA')['IMPORTO']
                                .rolling(window=window, min_periods=1).mean()
                                .reset_index().groupby('CATEGORIA')
                                .last().drop(['level_1'], axis=1))
            st.dataframe(category_expenses)  
        else:
            category_expenses = (transaction_df[(transaction_df['TIPO TRANSAZIONE'] == 'Uscita') & (transaction_df['mese_anno'] == ym1)]
                            .groupby(['mese_anno', 'CATEGORIA'])['IMPORTO']
                            .sum())
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
        choice1 = st.selectbox("Select the category for compute the monthly expenses", exit_category,  key="category_selectbox")
        st.write(f'<p style="font-size:24px; color:white;">Monthly expenditure chart for the {choice1} category</p>', 
                unsafe_allow_html=True)
        # plot the monthly expenses for category
        choice1_montly_exp=transact_plot.montly_ma_plot(column_name='IMPORTO', window=window, date_name = 'mese_anno', 
                                                        category=choice1, category_column='CATEGORIA')
        st.plotly_chart(choice1_montly_exp)

    with col2:
        subcategory = transaction_df.loc[transaction_df['TIPO TRANSAZIONE'] == 'Uscita', 'SOTTOCATEGORIA'].unique().tolist()
        subcategory = list(filter(pd.notna, subcategory))
        choice2 = st.selectbox("Select the subcategory for compute the monthly expenses", subcategory,  key="subcategory_selectbox")
        st.write(f'<p style="font-size:24px; color:white;">Monthly expenditure chart for the {choice2} subcategory</p>', 
                unsafe_allow_html=True)
        # plot the monthly expenses for category
        choice2_montly_exp=transact_plot.montly_ma_plot(column_name='IMPORTO', window=window, date_name = 'mese_anno', sub_category=choice2)
        st.plotly_chart(choice2_montly_exp)

    # daily expenses analysis
    col1, col2 = st.columns(2)
    with col1:
        st.write(f'<p style="font-size:24px; color:white;">Table of daily average expenses</p>', 
                unsafe_allow_html=True)
        daily_expenses_df = (transaction_df.dropna(subset=['GIORNO'])
                             .groupby('GIORNO')['IMPORTO']
                             .mean())
        st.dataframe(daily_expenses_df)
    
    with col2:
        fig = px.bar(
            daily_expenses_df.reset_index(), 
            x='GIORNO', 
            y='IMPORTO', 
            title='Barplot of Average daily expenses', 
            labels={'IMPORTO': 'Value (€)', 'GIORNO': 'Day'},  
            color='GIORNO',  
            text='IMPORTO' 
        )

        # add layout settings
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside') 
        fig.update_layout(
            uniformtext_minsize=8, 
            uniformtext_mode='hide',
            xaxis_title="Day",
            yaxis_title="Value (€)",
        )
        st.plotly_chart(fig)


# analysis of income
if page == 'Income':
    # title of the page
    st.title("Analysis of expenses") 
    col1, col2 = st.columns(2)

    with col1:
         st.write('<p style="font-size:24px; color:white;">Monthly income time series</p>', unsafe_allow_html=True)
         income_plot = transact_plot.montly_ma_plot(column_name='IMPORTO', window=window, date_name='mese_anno', 
                                                    type_transaction_col='TIPO TRANSAZIONE', type_transaction='Entrata')
         st.plotly_chart(income_plot)

    with col2: 
        year_month = (
            transaction_df
            .sort_values(by = 'mese_anno', ascending = False)['mese_anno']
            .astype('str')
            .unique()
        )
        year_month = list(year_month[:3]) + ['Mooving average']
        my = st.selectbox("Select between the month of interest or a mooving average", year_month)
        if my != 'Mooving average':
            income_df = (
            transaction_df[(transaction_df['TIPO TRANSAZIONE']=='Entrata') & (transaction_df['mese_anno']==my)]
            .groupby('CATEGORIA')[['IMPORTO']]
            .sum()
            )
        else:
            income_df = (
                transaction_df[transaction_df['TIPO TRANSAZIONE']=='Entrata']
                .groupby(['mese_anno', 'CATEGORIA'])['IMPORTO']
                .sum()
                .reset_index()
                .groupby('CATEGORIA')['IMPORTO']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index().groupby('CATEGORIA')
                .last().drop(['level_1'], axis=1)
                )
        st.write('<p style="font-size:24px; color:white;">Income divided by type</p>', unsafe_allow_html=True)
        st.dataframe(income_df)
        
    #Pie plot of income divided by type
    income_df.reset_index(inplace=True)
    st.write('<p style="font-size:24px; color:white;">Pie plot of income divided by type</p>', 
                 unsafe_allow_html=True)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=income_df['CATEGORIA'],        # Labels (categories)
                values=income_df['IMPORTO'],           # Values
                hoverinfo='label+percent+value', # Show label, percentage, and value on hover
                textinfo='percent',           # Display only percentage on the pie chart
                #marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']),  # Custom colors
                )
            ]
    )
    # Update layout to include a legend
    fig.update_layout(
        title="Income Category Distribution",
        legend=dict(title="Categories", orientation="v", x=1, y=1),
    )
    st.plotly_chart(fig)
        


