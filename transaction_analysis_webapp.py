from const import tipo_conto_ass, categories_associations, cols_to_keep
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transaction_exploration import graphical_analysis
import plotly.graph_objects as go
import plotly.express as px
from get_financial_data import get_historical_data
from datetime import datetime


path = '/Users/lucariotto/Documents/Personal/Gestione denaro/Analisi spese/Gestione entrate-spese.xlsx'
transaction_df = pd.read_excel(path, sheet_name='Transazioni')

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
exit_category = tipo_conto_ass['Uscita']
ciroconti_category = tipo_conto_ass['Giroconto']
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
                            .groupby(['mese_anno', 'CATEGORIA'])[['IMPORTO']]
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
        

# assets analysis
assets_df = pd.read_excel(path, sheet_name='Patrimonio')
# expand data to obtain daily frequencies for all assets
dates = list(assets_df['DATA'].unique()) + [pd.Timestamp.now()]
asset_df_exp = pd.DataFrame(columns=assets_df.columns)
for i in range(0, len(dates)-1):
    dt_corr = dates[i]
    dates_sup = dates[i+1] - pd.Timedelta(days=1)
    date_seq = pd.DataFrame({'DATA' : pd.date_range(start=dt_corr, end=dates_sup, freq='D')})
    assets_date_cor = (
        date_seq
        .merge(
            assets_df[assets_df['DATA']==dt_corr],
            how = 'cross'
        )
    )
    asset_df_exp = pd.concat([asset_df_exp, assets_date_cor], axis=0)

# download financial data
ticker = assets_df.dropna(subset=['TICKER'])['TICKER'].unique().tolist()
fin_df = get_historical_data(ticker, '2019-01-01', pd.Timestamp.now())

# expand data and calculate daily portfolio value
asset_df_exp = (
    asset_df_exp
    .drop(columns=['DATA', 'DATA_y'])
    .rename(columns={'DATA_x' : 'DATA'})
)
dta_df = pd.DataFrame({'DATA' : asset_df_exp["DATA"].unique()})
fin_df = (
    dta_df
    .merge(
        fin_df['Close']
        .reset_index()
        .rename(columns={'Date' : 'DATA'}),
        on='DATA',
        how='left')
)
fin_df[ticker] = fin_df[ticker].apply(lambda col: col.fillna(method='ffill'))

fin_data = (
    fin_df
    .drop(columns=['DATA'])
    .sort_index(axis=1)
    .values
)
max_length = len(ticker)
quantity_df = (
    dta_df
    .merge(
        pd.DataFrame(
            asset_df_exp
            .dropna(subset=['TICKER'])['TICKER']
            .unique(), 
            columns=['TICKER']
        ),
        how='cross'
    )
    .merge(
        asset_df_exp[['DATA','TICKER', 'QUANTITA']]
        .dropna(subset=['TICKER']),
        on=['DATA', 'TICKER'], how='left'
    )
    .fillna(0)
    .sort_values(['DATA', 'TICKER'])
    .groupby('DATA')
    .agg({'QUANTITA': list})['QUANTITA']
)

value_df = pd.DataFrame({
    'DATA' : dta_df['DATA'],
    'value' : np.diag(fin_data @ np.array(quantity_df.tolist()).T)}
)

# calculate portfolio vaue for every date
asset_df_exp = (
    asset_df_exp[asset_df_exp['TICKER'].isna()]
    .pivot_table(values='QUANTITA', index='DATA', columns='VOCE', aggfunc='sum', fill_value=0)
    .reset_index()
    .assign(QUANTITA=lambda x: x['Attivita']-x['Passivita'])
    .merge(value_df, on = 'DATA', how = 'outer')
    .fillna(0)
    .assign(curval=lambda x: x['value']+x['QUANTITA'])
    .drop(columns=['value', 'QUANTITA', 'Attivita', 'Passivita'])
)

# calculate nominal portfolio value give by the total value at the first date 
# and the cumsum of cashflow at every date 
min_date = min(asset_df_exp['DATA'])
asset_init = asset_df_exp[asset_df_exp['DATA'] == min_date]['curval'].values
cashflow_df = (
    dta_df
    .merge(
        (transaction_df
        .pivot_table(values='IMPORTO', index='DATA', columns='TIPO TRANSAZIONE', aggfunc='sum')
        .reset_index()
        .loc[lambda df: df['DATA'] >= min_date]
        .fillna(0)
        .assign(cashflow=lambda x: x['Entrata']-x['Uscita'])),
        on='DATA',
        how='left'
    )
    .sort_values('DATA')
    .apply(lambda col: col.fillna(0) if col.name =='cashflow' else col)
)
cashflow_df.loc[cashflow_df['DATA'] == min_date, 'cashflow'] += asset_init
cashflow_df['cashflow'] = cashflow_df['cashflow'].cumsum()

if page == 'Assets':
    st.title("Assets analysis")

    col1, col2 = st.columns(2)
    # date of time series
    with col1:
        start_date = pd.to_datetime(st.date_input("Select the start date", value=min_date))
    with col2:
        end_date = pd.to_datetime(st.date_input("Select the end date", value=datetime.now().date()))

    # Validity check for the dates
    if start_date > end_date:
        st.error("The start date cannot be later than the end date. Please correct the dates.")
    elif end_date > pd.Timestamp.now():
        st.error("The end date cannot be later than the current date. Please correct the dates.")
    else:
        st.success(f"Selected range: from {start_date} to {end_date}")
    
     # create plot object
    fig = go.Figure()
    # add line for nominal portfolio value
    fig.add_trace(go.Scatter(
        x=cashflow_df.loc[(cashflow_df['DATA']>=start_date) & (cashflow_df['DATA']<=end_date), 'DATA'], 
        y=cashflow_df.loc[(cashflow_df['DATA']>=start_date) & (cashflow_df['DATA']<=end_date), 'cashflow'], 
        mode='lines',
        name='Nominal portfolio value',
        line=dict(color='blue'),
        marker=dict(size=8)
    ))

    # add line for real portfolio value
    fig.add_trace(go.Scatter(
        x=value_df.loc[(value_df['DATA']>=start_date) & (value_df['DATA']<=end_date), 'DATA'], 
        y=value_df.loc[(value_df['DATA']>=start_date) & (value_df['DATA']<=end_date), 'value'], 
        mode='lines',
        name='Real portfolio value',
        line=dict(color='red'),
        marker=dict(size=8)
    ))

    # custom layout
    fig.update_layout(
        title="Time Series of Portfolio Value",
        xaxis_title="Date",
        yaxis_title="",
        legend=dict(title="Legend", x=0.8, y=1),
        template="plotly_white",
        hovermode="x unified" 
    )
    
    st.plotly_chart(fig)

    # table and pie plot of asset class allocation
    max_dta = max(assets_df['DATA'])
    asset_alloc_no_tk = (
        assets_df[
            (assets_df['DATA'] == max_dta) &
            (assets_df['TICKER'].isna()) &
            (assets_df['VOCE'] == 'Attivita') 
        ]   
        .groupby(['DATA', 'ASSET CLASS'])
        .agg({'QUANTITA': sum})
        .reset_index()
    )
    ticker_melt = (
        fin_df[fin_df['DATA'] == max_dta]
        .melt(
            id_vars='DATA', 
            value_vars=assets_df.loc[(assets_df['DATA'] == max_dta) & (~assets_df['TICKER'].isna()), 'TICKER'].unique(), 
            var_name='TICKER',         
            value_name='VALUE'
        )
    )
    asset_alloc_tk = (
        assets_df.loc[
            (assets_df['DATA'] == max_dta) &
            (~assets_df['TICKER'].isna()) &
            (assets_df['VOCE'] == 'Attivita'),
            ['DATA', 'ASSET CLASS', 'TICKER', 'QUANTITA']
        ]
        .merge(
            ticker_melt,
            on=['DATA', 'TICKER'],
            how='left'
        )
        .assign(QUANTITA = lambda x: x['QUANTITA'] * x['VALUE'])
        .drop(['VALUE', 'TICKER'], axis=1)
    )
    asset_alloc_df = (
        pd.concat([asset_alloc_no_tk, asset_alloc_tk], axis=0)
        .groupby('ASSET CLASS')
        .sum("QUANTITA")
    )

    total_portfolio = sum(asset_alloc_df['QUANTITA'])
    asset_alloc_df['% ASSET ALLOCATION'] = round(asset_alloc_df['QUANTITA'] / total_portfolio * 100, 2)

    st.write('<p style="font-size:24px; color:white;">Asset class allocation</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(asset_alloc_df)

    with col2:
        asset_alloc_df.reset_index(inplace=True)
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=asset_alloc_df['ASSET CLASS'],  
                    values=asset_alloc_df['QUANTITA'],  
                    hoverinfo='label+percent+value', 
                    textinfo='percent',  
                    #marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']),  # Custom colors
                    )
                ]
        )
        # Update layout to include a legend
        fig.update_layout(
            title="Asset Class Distribution",
            legend=dict(title="Categories", orientation="v", x=1, y=1),
        )
        st.plotly_chart(fig)