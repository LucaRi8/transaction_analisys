
import pandas as pd
import requests

# Function to download historical data for a specific stock
def get_historical_data(ticker, from_date, to_date, API_KEY):
    """
    Download historical data from Polygon.io for a specific stock.

    Args:
        ticker (str): Stock symbol (e.g., "AAPL" for Apple).
        from_date (str): Start date (format: "YYYY-MM-DD").
        to_date (str): End date (format: "YYYY-MM-DD").

    Returns:
        pandas.DataFrame: A dataframe containing the historical data.
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
    params = {
        "apiKey": API_KEY
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        
        # Convert the results into a DataFrame
        df = pd.DataFrame(results)
        if not df.empty:
            # Convert timestamps and organize columns
            df['t'] = pd.to_datetime(df['t'], unit='ms')  # Convert the timestamp
            df.rename(columns={
                't': 'date',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            }, inplace=True)
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        return df
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
