# import library 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# creation of a class for all grapichal analysis of transaction
class graphical_analysis:
    def __init__(self, df):
        self.df = df

    def montly_ewma_plot(self, column_name, alpha = 0.1, date_name = 'DATE', category = 'all', category_column = 'CATEGORIA',
                         type_transaction_col = 'TIPO TRANSAZIONE', type_transaction = 'Uscita', sub_category = 'all', 
                         sub_category_column = 'SOTTOCATEGORIA'):
        # function that perform an aggregation over month-year of the column and
        # plot the raw data and the ewma mean of the data over time
        # column_name : the name of the column to plot
        # date_name : the name of the column that contain the date, if missing date_name = 'DATE'
        # alpha : the value of the parameter of the ewma statistic
        # category : set the specific category to filter data
        # category_column : define the name of the column thath contain the different category of expenses
        # sub_category : set the specific sub category to filter data
        # sub_category_column : define the name of the column thath contain the different sub category of expenses
        # type_transaction_col : col that define the type of transaction 
        # type_transaction : the type of transaction income or expenses

        if type_transaction_col not in self.df.columns:
            raise ValueError(f"type_transaction_col must be one of {[col for col in self.df.columns if col != date_name]}")
        if type_transaction not in self.df[type_transaction_col].unique().tolist():
            raise ValueError(f"Type transaction must be one of {self.df[type_transaction_col].unique().tolist()}")
            
        self.df = self.df[self.df[type_transaction_col] == type_transaction]

        if category != 'all':
            if category_column not in self.df.columns:
                raise ValueError(f"category_column must be one of {[col for col in self.df.columns if col != date_name]}")
            if category not in self.df[category_column].unique().tolist():
                raise ValueError(f"category must be one of {self.df[category_column].unique().tolist()}")
            self.df = self.df[self.df[category_column] == category]
        
        if sub_category != 'all':
            if sub_category_column not in self.df.columns:
                raise ValueError(f"sub category_column must be one of {[col for col in self.df.columns if col != date_name]}")
            if sub_category not in self.df[sub_category_column].unique().tolist():
                raise ValueError(f"sub category must be one of {self.df[sub_category_column].unique().tolist()}")
            self.df = self.df[self.df[sub_category_column] == sub_category]

        self.df.loc[:,'mese_anno'] = self.df[date_name].dt.to_period('M')
        self.df = self.df.groupby('mese_anno')[[column_name]].sum()
        self.df.reset_index(inplace=True)
        self.df['mese_anno'] = self.df['mese_anno'].astype('str')
        self.df[f'{column_name}_ewma_05'] = self.df[column_name].ewm(alpha=0.1, adjust=False).mean()

        # plot the montly aggregation
        plt.figure(figsize=(10, 6))  # Impostare le dimensioni del grafico
        plt.plot(self.df["mese_anno"], self.df[f"{column_name}_ewma_05"], 
                marker="o", linestyle="-", color="b", label=f'ewma of montly {column_name}')
        plt.plot(self.df["mese_anno"], self.df[column_name], 
                marker="o", linestyle="-", color="r", label=[f'raw montly {column_name}'])
        plt.xlabel("")
        plt.ylabel(f"monthly {column_name}")
        plt.title(f"Time series of monthly {column_name}")
        plt.legend(loc='best')
        plt.show()

    def cash_flow_plot(self, column_name, date_name = 'DATE', alpha=0.1, type_transaction_col = 'TIPO TRANSAZIONE'):
        # function that perform an aggregation over month-year of the column and
        # plot the the difference between income and expence
        # column_name : the name of the column to plot
        # date_name : the name of the column that contain the date, if missing date_name = 'DATE'
         # alpha : the value of the parameter of the ewma statistic
        # type_transaction_col : col that define the type of transaction 
        if type_transaction_col not in self.df.columns:
            raise ValueError(f"type_transaction_col must be one of {[col for col in self.df.columns if col != date_name]}")
        
        self.df.loc[:,'mese_anno'] = self.df[date_name].dt.to_period('M')
        pivot_df = self.df.pivot_table(values=column_name, index='mese_anno', columns=type_transaction_col, aggfunc='sum')
        pivot_df.reset_index(inplace=True)
        pivot_df['cash_flow'] = pivot_df['Entrata'] - pivot_df['Uscita']
        pivot_df['ewma_cash_flow'] = pivot_df['cash_flow'].ewm(alpha=0.1, adjust=False).mean()
        pivot_df['mese_anno'] = pivot_df['mese_anno'].astype('str')
        # plot the montly aggregation
        plt.figure(figsize=(10, 6))  # Impostare le dimensioni del grafico
        plt.plot(pivot_df["mese_anno"], pivot_df["ewma_cash_flow"], 
                marker="o", linestyle="-", color="b", label='ewma_cash_flow')
        plt.plot(pivot_df["mese_anno"], pivot_df['cash_flow'], 
                marker="o", linestyle="-", color="r", label='cash_flow')
        plt.xlabel("")
        plt.ylabel("monthly cash_flow")
        plt.title("Time series of monthly cash_flow")
        plt.legend(loc='best')
        plt.show()
        



path = '/Users/lucariotto/Documents/Personal/Gestione denaro/Analisi spese/Gestione entrate-spese.xlsx'
transaction_df = pd.read_excel(path, sheet_name='Transazioni')
gr = graphical_analysis(transaction_df)
gr.cash_flow_plot(column_name='IMPORTO', alpha=0.1, date_name='DATA')