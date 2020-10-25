import pandas as pd

# Read in dataset
index_prices = pd.read_csv('sphist.csv')

# Convert Date column to date type
index_prices['Date'] = pd.to_datetime(index_prices['Date'])

# Convert all column names to snake_casing
index_prices.columns = index_prices.columns.str.lower().str.replace(' ', '_')

# Import datetime module 
from datetime import datetime

# Sort dataframe by Date in ascending order
index_prices = index_prices.sort_values(by = 'date')


# Some indicators that are interesting to generate for each row:
# * The average price from the past 5 days.
# * The average price for the past 30 days.
# * The average price for the past 365 days.
# * The ratio between the average price for the past 5 days, and the average price for the past 365 days.
# * The standard deviation of the price over the past 5 days.
# * The standard deviation of the price over the past 365 days.
# * The ratio between the standard deviation for the past 5 days, and the standard deviation for the past 365 days.

# Pick 3 indicators to compute, and generate a different column for each one.


# Create a copy of index_prices as historical_prices
historical_prices = index_prices.copy()


# Create a function to generate average price for the past n days
def past_n_avg(n):
    return index_prices.close.rolling(n).mean().shift(periods = 1)

# Generate average price for the past 5 days.
index_prices['past_5_avg'] = past_n_avg(5) 

# Generate average price for the past 30 days.
index_prices['past_30_avg'] = past_n_avg(30)

# Generate average price for the past 365 days.
index_prices['past_365_avg'] = past_n_avg(365)

# Generate the ratio between the average price for the past 5 days, and the average price for the past 365 days.
index_prices['past_5_365_ratio'] = index_prices.past_5_avg/index_prices.past_365_avg

# Generate the standard deviation of the price over the past 5 days.
index_prices['past_5_std'] = index_prices.close.rolling(5).std().shift(periods = 1)

# Some of the indicators use 365 days of historical data, and the dataset starts on 1950-01-03. Thus, any rows that fall before 1951-01-03 don't have enough historical data to compute all the indicators.

# Remove entries that are earlier than 1951-01-03 
index_prices = index_prices[index_prices.date > datetime(year=1951, month=1, day=2)]

# Remove any rows with missing values 
index_prices.dropna(inplace = True)

print(index_prices.head(10), '\n', '*'*30)

# Generate two dataframes, one for training, one for testing
cutoff = datetime(year = 2013, month = 1, day = 1)
train = index_prices[index_prices.date < cutoff]
test = index_prices[index_prices.date >= cutoff]

print(train.shape, test.shape)

# Import Linear Regression model 
from sklearn.linear_model import LinearRegression
# Use Mean Absolute Error (MAE) as error metric 
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

features = ['past_5_avg', 'past_30_avg', 'past_365_avg', 'past_5_365_ratio', 'past_5_std']

for i in range(len(features)):
    lr = LinearRegression()
    lr.fit(train[features[:i+1]], train.close)
    preds = lr.predict(test[features[:i+1]])
    mae = mean_absolute_error(test.close, preds)
    print('With features: {}, the mean absolute error of the model is {}.'.format(features[:i+1], mae))



for alpha in range(1,10):
    ridge = Ridge(alpha = alpha)
    ridge.fit(train[features], train.close) 
    preds = ridge.predict(test[features])
    mae = mean_absolute_error(test.close, preds)
    print('With an alpha {}, the mean absolute error of the model is {}.'.format(alpha, mae))




