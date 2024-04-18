import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

# full=True
file_name = "2000-2023"
df = pd.read_csv(file_name + ".csv")
print(df)
print(df.columns)

# 1. Data clean

# Select needed columns
df = df.loc[:, ['PERMNO', 'SecurityBegDt', 'SecurityEndDt', 'TradingStatusFlg', 'SICCD', 'YYYYMMDD', 'DlyCalDt',
       'DlyPrc', 'DlyCap', 'DlyPrevPrc', 'DlyPrevCap', 'DlyRet', 'DlyRetx', 'DlyRetMissFlg',
       'DlyFacPrc', 'DlyVol', 'DlyClose', 'DlyLow', 'DlyHigh',
       'DlyOpen', 'DlyPrcVol', 'ShrOut']]

# Drop duplicate data
df.drop_duplicates(keep='first', inplace=True)

print(df[df.duplicated(subset=['PERMNO', 'DlyCalDt'])])

# Filter out stock listing day's data
df = df[df['SecurityBegDt'] != df['DlyCalDt']]

# Select dates with normal trading status
df = df[df['TradingStatusFlg'] == 'A']

# Select dataes with return data
df = df[df['DlyRetMissFlg'].isna()]

# CRSP's volume-related data were all divided by 1000, need to readjust to normal number
for col in ['DlyCap', 'DlyVol', 'DlyPrcVol', 'ShrOut']:
    df[col] = df[col] * 1000

# Select only data with positive trading volume
df = df[(df['DlyVol'] != 0.0) & (~df['DlyVol'].isna())]
 
# Filter out data without basic price data
price_filter = df['DlyClose'].isna() & df['DlyLow'].isna() & df['DlyHigh'].isna() & df['DlyOpen'].isna()
df = df[~price_filter]

# Filter out data without market cap features
mkt_cap_filter = df['ShrOut'].isna() & df['DlyCap'].isna()
df = df[~mkt_cap_filter]

# For missing price data, we use other price from current or previous day to replace them
df['DlyOpen'] = df['DlyOpen'].fillna(df['DlyPrevPrc'] / df['DlyFacPrc'])

max_values = df[['DlyClose', 'DlyOpen']].max(axis=1)
df['DlyHigh'] = df['DlyHigh'].fillna(max_values)

min_values = df[['DlyClose', 'DlyOpen']].min(axis=1)
df['DlyLow'] = df['DlyLow'].fillna(min_values)


# Select useful data for creating features
df = df.loc[:, ['PERMNO', 'SecurityEndDt', 'SICCD', 'DlyCalDt',
       'DlyPrc', 'DlyCap', 'DlyPrevPrc', 'DlyRet',
       'DlyFacPrc', 'DlyVol', 'DlyClose', 'DlyLow', 'DlyHigh',
       'DlyOpen', 'DlyPrcVol', 'ShrOut']]

print(df[df.isna().any(axis=1)])


# if not full:
#     df= df[(df['SecurityBegDt'] <= '1999-12-01') & (df['SecurityEndDt'] >= '2017-12-31')]
#     df.sort_values(by=['PERMNO', 'DlyCalDt'], inplace=True)
#     # print(filter)
#     # gb = filter.groupby('PERMNO')
#     # df_list = []
#     # for name, group in gb:
#     #     if group['DlyCalDt'].to_numpy()[-1] < '2017-12-01':
#     #         continue
#     #     df_list.append(group)
#     #
#     # df = pd.concat(df_list)
#     df = df[df['DlyCalDt'] > '1999-12-01']
#     count = df.groupby('PERMNO').count()
#     mode = count['Ticker'].mode().to_numpy()[0]
#     count = count[count['Ticker'] == mode]

#     df = df[df['PERMNO'].isin(count.index)]
# else:
#     df = df[(df['DlyCalDt'] > '1999-12-01') & (df['DlyCalDt'] <= '2017-12-31')]

# 2. Create features
gb = df.groupby('PERMNO')

df_list = []

for name, group in gb:
    # prev_low = group['DlyLow'].to_numpy()[:-1]
    # prev_high = group['DlyHigh'].to_numpy()[:-1]
    # prev_open = group['DlyOpen'].to_numpy()[:-1]
    
    vol = group['DlyVol'].pct_change().to_numpy()[1:]
    amount = group['DlyPrcVol'].pct_change().to_numpy()[1:]
    group = group.iloc[1:]
    # factor = group['DlyFacPrc'].to_numpy()
    # group['prev_low'] = prev_low / factor
    # group['prev_high'] = prev_high / factor
    # group['prev_open'] = prev_open / factor
    group['close'] = (group['DlyClose'] - group['DlyOpen']) / group['DlyOpen']
    group['high'] = (group['DlyHigh'] - group['DlyOpen']) / group['DlyOpen']
    group['low'] = (group['DlyLow'] - group['DlyOpen']) / group['DlyOpen']
    group['open'] = (group['DlyOpen'] - group['DlyPrevPrc']) / group['DlyOpen']
    group['volume'] = vol
    group['amount'] = amount
    label = group['DlyRet'].to_numpy()[1:]
    group = group.iloc[:-1]
    group['label'] = label
    
    group.fillna(0, inplace=True)
    group.replace([np.inf, -np.inf], 0, inplace=True)
    group = group.loc[:, ['PERMNO', 'SecurityEndDt', 'SICCD', 'DlyCalDt', 'DlyRet', 'high', 'low', 'open', 'close',
                        'volume', 'amount', 'label', 'DlyPrc', 'ShrOut', 'DlyVol']]
    group.rename(columns={'DlyRet': 'return', 'DlyCalDt': 'date'}, inplace=True)
    df_list.append(group)

df = pd.concat(df_list)


# 3. Feature clean
features = ['open', 'high', 'low', 'close', 'return', 'volume', 'amount']
print(df.loc[:, features].describe())

# Remove outliers from data
def remove_outliers_zscore(df, columns, threshold=3):
    df_no_outliers = df.copy()
    for col in columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df_no_outliers = df_no_outliers[(z_scores < threshold)]
    return df_no_outliers

def remove_outliers_iqr(df, columns):
    df_no_outliers = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df_no_outliers = df_no_outliers[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    return df_no_outliers


def remove_outliers_mad(df, columns, threshold=3):
    df_no_outliers = df.copy()
    for col in columns:
        median_value = df[col].median()
        absolute_deviations = np.abs(df[col] - median_value)
        mad = np.median(absolute_deviations)
        outliers = df[col][absolute_deviations > threshold * mad]
        df_no_outliers = df_no_outliers[~df_no_outliers[col].isin(outliers)]
    return df_no_outliers

df = remove_outliers_zscore(df, features)

print(df.loc[:, features].describe())

col_norm = [x + "_norm"  for x in features]

# Standardize factors by daily frequency
gb = df.groupby("date")

df_list = []
scaler = RobustScaler()
for name, group in gb:
    group[col_norm] = scaler.fit_transform(group[features])
    df_list.append(group)

df = pd.concat(df_list)

df.to_feather(file_name + "_normalized_feature.feather")