import requests
import pandas as pd

def get_historical_data(api_key, symbol, function="TIME_SERIES_DAILY_ADJUSTED", outputsize="compact"):
    """
    Download financial data from Alpha Vantage.

    Parameters:
        api_key (str): The Alpha Vantage API key.
        symbol (str): The stock symbol to download (e.g., "AAPL" for Apple).
        function (str): The type of data to download. Default: "TIME_SERIES_DAILY_ADJUSTED".
        outputsize (str): Data size ("compact" or "full"). Default: "compact".

    Returns:
        pd.DataFrame: A DataFrame containing the financial data.
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": outputsize,
        "datatype": "json"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract the relevant data section from the JSON
        time_series_key = "Time Series (Daily)" if function == "TIME_SERIES_DAILY_ADJUSTED" else None
        if not time_series_key or time_series_key not in data:
            raise ValueError("Unrecognized data format or missing key in JSON.")

        time_series = data[time_series_key]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        df.index = pd.to_datetime(df.index)  # Convert the index to datetime format

        # Rename columns for clarity
        df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adjusted_close",
            "6. volume": "volume",
            "7. dividend amount": "dividend_amount",
            "8. split coefficient": "split_coefficient"
        }, inplace=True)

        # Convert values to numeric where possible
        df = df.apply(pd.to_numeric, errors="coerce")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except ValueError as e:
        print(f"Data error: {e}")
        return None
