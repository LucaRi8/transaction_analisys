# import library 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# creation of a class for all grapichal analysis of transaction
class graphical_analysis:
    def __init__(self, df):
        self.df = df

    def montly_ewma_plot(self, column_name, alpha, date_name = 'DATE', category = 'all', category_column = 'CATEGORIA',
                         sub_category = 'all', sub_category_column = 'SOTTOCATEGORIA'):
        # function that perform an aggregation over month-year of the column and
        # plot the raw data and the ewma mean of the data over time
        # column_name : the name of the column to plot
        # date_name : the name of the column that contain the date, if missing date_name = 'DATE'
        # alpha : the value of the parameter of the ewma statistic
        # category : set the specific category to filter data
        # category_column : define the name of the column thath contain the different category of expenses
        # sub_category : set the specific sub category to filter data
        # sub_category_column : define the name of the column thath contain the different sub category of expenses

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

        # Creare il grafico
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


gr = graphical_analysis(transaction_df)
gr.montly_ewma_plot(column_name='IMPORTO', alpha=0.1, date_name='DATA', category='Casa')