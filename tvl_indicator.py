import requests
import json
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import tabulate


# Get TVL data from defillama
req = requests.get('https://api.llama.fi/charts/Ethereum')
req_json = json.loads(req.text)

tvl_df = pd.DataFrame(req_json)
tvl_df['date'] = tvl_df['date'].astype(int)

# Load price data
bars = pd.read_csv('ETHUSDT86400.csv')
bars = bars.set_index('open_time')

# Shift date column so data is concurrent, bars has the open time
tvl_df['date'] = tvl_df['date'].shift(1)
tvl_df = tvl_df.set_index('date')

# Add tvl to bars
bars['tvl'] = tvl_df['totalLiquidityUSD']

# Convert epoch time to human time (UTC)
bars['datetime'] = pd.to_datetime(bars.index, unit='s')
bars = bars.set_index('datetime')


# Plot TVL and Close
'''
fig, ax1 = plt.subplots()
ax1.set_xlabel('Date')
ax1.set_ylabel('close', color='tab:green')
ax1.tick_params(axis='y', labelcolor='tab:green')
ax1.plot(bars.index, bars['close'], color='tab:green')

ax2 = ax1.twinx()
ax2.set_ylabel('TVL', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.plot(bars.index, bars['tvl'], color='tab:blue')

fig.tight_layout()
plt.title("Close and TVL")
plt.show()

# Plot Scatter
bars.plot.scatter('close', 'tvl')
plt.title("Close Vs. TVL Scatter")
plt.show()
'''
# Estimate current price using TVL
def rolling_fit(df: pd.DataFrame, x_col, y_col, window):
    pred = [np.nan] * df.shape[0]
    for i in range(window - 1, df.shape[0]):
        x_slice = df[x_col].iloc[i - window + 1: i+1]
        y_slice = df[y_col].iloc[i - window + 1: i+1]
        
        x_slice = np.log(x_slice)
        y_slice = np.log(y_slice)

        coefs = np.polyfit(x_slice, y_slice, 1)
        pred[i] = coefs[0] * x_slice.iloc[-1] + coefs[1]

        pred[i] = np.exp(pred[i])
        
    return pred

# Compute indicator
bars = bars.dropna()
fit_length = 7 # Main parameter
bars['pred'] = rolling_fit(bars, 'tvl', 'close', fit_length)
bars = bars.dropna()


def atr(df, window):
    data = df.copy()
    high = data['high']
    low = data['low']
    close = data['close']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

bars['atr'] = atr(bars, 30) # Volatility normalization, should be >= than fit_length
bars = bars.dropna()
bars['ind'] = (bars['close'] - bars['pred']) / bars['atr']

# Optionally apply normal CDF, compresses outliers and makes the indicator have hard bounds at -1 and 1.
# IF you were to feed this indicator to a predictive model, you would likely want to do this. 
#bars['ind_cdf'] = 2.0 * scipy.stats.norm.cdf(bars['ind']) - 1.0

# Plot price and price predicted from TVL
'''
bars['close'].plot(label='Price')
bars['pred'].plot(label='TVL Pred Price')
plt.title("Price and TVL Predicted Price")
plt.legend()
plt.show()
# Plot indicator
fig, ax1 = plt.subplots()
ax1.set_xlabel('Date')
ax1.set_ylabel('close', color='tab:green')
ax1.tick_params(axis='y', labelcolor='tab:green')
ax1.plot(bars.index, bars['close'], color='tab:green')

ax2 = ax1.twinx()
ax2.set_ylabel('TVL Indicator', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.plot(bars.index, bars['ind'], color='tab:blue')

fig.tight_layout()
plt.title("Close and TVL Indicator")
plt.show()
'''

# Compute next days percent return
bars['next_return'] = bars['close'].pct_change().shift(-1)
bars['next_return'] *= 100

# Get performance correlation and above/below thersholds
results_df = pd.DataFrame()
for year in ['all', '2019', '2020', '2021', '2022']:
    data = None
    if year == 'all':
        data = bars
    else:
        data = bars[bars.index.year == int(year)]

    # Spearman Correlation
    results_df.loc[year, 'Spearman Corr'] = round(data['next_return'].corr(data['ind'], method='spearman'), 3)
    results_df.loc[year, 'Ind > 0'] = f"{round(data[data['ind'] > 0]['next_return'].mean(), 3)}%"
    results_df.loc[year, 'Ind < 0'] = f"{round(data[data['ind'] < 0]['next_return'].mean(), 3)}%"
    results_df.loc[year, 'Ind >  0.25'] = f"{round(data[data['ind'] > 0.25]['next_return'].mean(), 3)}%"
    results_df.loc[year, 'Ind < -0.25'] = f"{round(data[data['ind'] < -0.25]['next_return'].mean(), 3)}%"
    results_df.loc[year, 'Ind > 0.5'] = f"{round(data[data['ind'] > 0.5]['next_return'].mean(), 3)}%"
    results_df.loc[year, 'Ind < -0.5'] = f"{round(data[data['ind'] < -0.5]['next_return'].mean(), 3)}%"


print(tabulate.tabulate(results_df, headers=results_df.columns))





