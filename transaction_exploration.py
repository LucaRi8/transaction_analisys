# import library 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# creation of a class for all grapichal analysis of transaction
class graphical_analysis:
    def __init__(self, df):
        self.df = df

    def montly_ma_plot(self, column_name, window = 12, date_name = 'mese_anno', category = 'all', category_column = 'CATEGORIA',
                         type_transaction_col = 'TIPO TRANSAZIONE', type_transaction = 'Uscita', sub_category = 'all', 
                         sub_category_column = 'SOTTOCATEGORIA'):
        # function that perform an aggregation over month-year of the column and
        # plot the raw data and the ewma mean of the data over time
        # column_name : the name of the column to plot
        # date_name : the name of the column that contain the date, if missing date_name = 'mese_anno'
        # window : the value of the parameter of the mooving average
        # category : set the specific category to filter data
        # category_column : define the name of the column thath contain the different category of expenses
        # sub_category : set the specific sub category to filter data
        # sub_category_column : define the name of the column thath contain the different sub category of expenses
        # type_transaction_col : col that define the type of transaction 
        # type_transaction : the type of transaction income or expenses

        montly_ma_df = self.df
        if type_transaction_col not in montly_ma_df.columns:
            raise ValueError(f"type_transaction_col must be one of {[col for col in montly_ma_df.columns if col != date_name]}")
        if type_transaction not in montly_ma_df[type_transaction_col].unique().tolist():
            raise ValueError(f"Type transaction must be one of {montly_ma_df[type_transaction_col].unique().tolist()}")
            
        montly_ma_df = montly_ma_df[montly_ma_df[type_transaction_col] == type_transaction]

        if category != 'all':
            if category_column not in montly_ma_df.columns:
                raise ValueError(f"category_column must be one of {[col for col in montly_ma_df.columns if col != date_name]}")
            if category not in montly_ma_df[category_column].unique().tolist():
                raise ValueError(f"category must be one of {montly_ma_df[category_column].unique().tolist()}")
            montly_ma_df = montly_ma_df.groupby([date_name, category_column])[[column_name]].sum().reset_index()
            montly_ma_df = montly_ma_df[montly_ma_df[category_column] == category]
        
        if sub_category != 'all':
            if sub_category_column not in montly_ma_df.columns:
                raise ValueError(f"sub category_column must be one of {[col for col in montly_ma_df.columns if col != date_name]}")
            if sub_category not in montly_ma_df[sub_category_column].unique().tolist():
                raise ValueError(f"sub category must be one of {montly_ma_df[sub_category_column].unique().tolist()}")
            montly_ma_df = montly_ma_df[montly_ma_df[sub_category_column] == sub_category]

        montly_ma_df = montly_ma_df.groupby(date_name)[[column_name]].sum()
        montly_ma_df.reset_index(inplace=True)
        montly_ma_df[date_name] = montly_ma_df[date_name].astype('str')
        montly_ma_df[f'{column_name}_ma'] = montly_ma_df[column_name].rolling(window=window, min_periods=1).mean()

        # create plot object
        fig = go.Figure()

        # add moving average line
        fig.add_trace(go.Scatter(
            x=montly_ma_df[date_name], 
            y=montly_ma_df[f"{column_name}_ma"], 
            mode='lines+markers',
            name='Moving average',
            line=dict(color='blue'),
            marker=dict(size=8)
        ))

        # add line for raw data
        fig.add_trace(go.Scatter(
            x=montly_ma_df[date_name], 
            y=montly_ma_df[column_name], 
            mode='lines+markers',
            name='Raw data',
            line=dict(color='red'),
            marker=dict(size=8)
        ))

        # custom layout
        fig.update_layout(
            title="Monthly Aggregation with Moving Average",
            xaxis_title="Year-Month",
            yaxis_title="Monthly Value",
            legend=dict(title="Legend", x=0.8, y=1),
            template="plotly_white",
            hovermode="x unified" 
        )

        return fig

    def cash_flow_plot(self, column_name, date_name = 'mese_anno', window=12, type_transaction_col = 'TIPO TRANSAZIONE',
                       transaction_type1 = 'Entrata', transaction_type2 = 'Uscita'):
        # function that perform an aggregation over month-year of the column and
        # plot the the difference between income and expence
        # column_name : the name of the column to plot
        # date_name : the name of the column that contain the date, if missing date_name = 'mese_anno'
        # window : the value of the parameter of the mooving average
        # type_transaction_col : col that define the type of transaction 
        # transaction_type1
        # transaction_type2

        cash_flow_df = self.df

        if type_transaction_col not in cash_flow_df.columns:
            raise ValueError(f"type_transaction_col must be one of {[col for col in cash_flow_df.columns if col != date_name]}")
        
        pivot_df = cash_flow_df.pivot_table(values=column_name, index=date_name, columns=type_transaction_col, aggfunc='sum')
        pivot_df.reset_index(inplace=True)
        pivot_df['cash_flow'] = pivot_df[transaction_type1] - pivot_df[transaction_type2]
        pivot_df['ma_cash_flow'] = pivot_df['cash_flow'].rolling(window=window, min_periods=1).mean()
        pivot_df[date_name] = pivot_df[date_name].astype('str')
        
        # create plot object
        fig = go.Figure()
        # add line for mooving average of cash_flow
        fig.add_trace(go.Scatter(
            x=pivot_df[date_name], 
            y=pivot_df["ma_cash_flow"], 
            mode='lines+markers',
            name='Moving average cash flow',
            line=dict(color='blue'),
            marker=dict(size=8)
        ))

        # add line for raw cashflow
        fig.add_trace(go.Scatter(
            x=pivot_df[date_name], 
            y=pivot_df['cash_flow'], 
            mode='lines+markers',
            name='Raw cash flow',
            line=dict(color='red'),
            marker=dict(size=8)
        ))

        # custom layout
        fig.update_layout(
            title="Time Series of Monthly Cash Flow",
            xaxis_title="Year-Month",
            yaxis_title="Monthly Cash Flow",
            legend=dict(title="Legend", x=0.8, y=1),
            template="plotly_white",
            hovermode="x unified" 
        )
            
        return fig
        