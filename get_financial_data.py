import yfinance as yf

def download_financial_data(ticker, start_date, end_date):
    """
    Downloads daily financial data for a given stock ticker between start and end dates.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL', 'GOOG', etc.).
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: A DataFrame containing the stock data.
    """
    # Download stock data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Check if data is empty
    if stock_data.empty:
        print(f"No data found for ticker: {ticker} in the given date range.")
        return None
    
    return stock_data
