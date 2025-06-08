import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


# Load CSV, skipping the first row which repeats the column names
df = pd.read_csv('med_sale_price.csv', encoding='utf-16', sep='\t', skiprows=1)

# Rename first column
df.rename(columns={df.columns[0]: 'Region'}, inplace=True)

# Melt wide to long format
df_long = df.melt(id_vars='Region', var_name='Date', value_name='Median_Price')

# Remove dollar signs and Ks
df_long['Median_Price'] = df_long['Median_Price'].replace(r'[\$,K]', '', regex=True)
df_long['Median_Price'] = pd.to_numeric(df_long['Median_Price'], errors='coerce') * 1000

# Convert dates
df_long['Date'] = pd.to_datetime(df_long['Date'], format='%B %Y', errors='coerce')  # Format: January 2012
df_long.dropna(inplace=True)

# Step 1: Filter for one region (Honolulu)
#region = 'Honolulu, HI'
#region_df = df_long[df_long['Region'] == region].sort_values('Date')


regions = ['Honolulu, HI', 'Aiea, HI', 'East Honolulu, HI']
plt.figure(figsize=(12, 6))

for region in regions:
    temp_df = df_long[df_long['Region'] == region]
    plt.plot(temp_df['Date'], temp_df['Median_Price'], label=region)

plt.title('Home Prices in 3 Hawaii Regions')
plt.legend()

plt.xlabel('Date (Years)')
plt.ylabel('Median Price ($)')
plt.tight_layout()
plt.show()



# Step 2: Turn dates into numeric values (days since the start)
region_df['Days'] = (region_df['Date'] - region_df['Date'].min()).dt.days
X = region_df[['Days']]
y = region_df['Median_Price']

# Step 3: Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict future prices for the next 365 days
last_day = region_df['Days'].max()
future_days = np.arange(last_day + 1, last_day + 366).reshape(-1, 1)
future_dates = pd.date_range(start=region_df['Date'].max() + pd.Timedelta(days=1), periods=365)

predicted_prices = model.predict(future_days)

# Step 5: Plot historical + predicted prices
# plt.figure(figsize=(12, 6))
# plt.plot(region_df['Date'], y, label='Historical Prices')
# plt.plot(future_dates, predicted_prices, label='Predicted Prices', linestyle='dashed')
# plt.title(f'{region} - Home Price Forecast (Linear Regression)')
# plt.xlabel('Date')
# plt.ylabel('Median Price ($)')
# plt.legend()
# plt.tight_layout()
# plt.show()

# Plot
# plt.figure(figsize=(12, 6))
# for region in df_long['Region'].dropna().unique():
#     region_data = df_long[df_long['Region'] == region]
#     if not region_data.empty:
#         plt.plot(region_data['Date'], region_data['Median_Price'], label=region)
#
# plt.title('Hawaii Median Home Sale Price Over Time')
# plt.xlabel('Date')
# plt.ylabel('Median Price ($)')
# plt.legend()
# plt.tight_layout()
# plt.show()






