# 1. Given a list of stock_prices in ascending order by datetime, and their respective dates in list dts, write a function max_profit that outputs the max profit by buying and selling at a specific interval.

# Example:

# stock_prices = [10,5,20,32,25,12]
# dts = [
#     '2019-01-01', 
#     '2019-01-02',
#     '2019-01-03',
#     '2019-01-04',
#     '2019-01-05',
#     '2019-01-06',
# ]

# Solution:
# Iterative approach, where we basically have 2 loops, first loop running from 0 to the second last stock date, and second loop running from 1 to the last date, if max_profit < diff
# we then set max_profit to diff and set buy and sell to i and j respectively to update the index

def get_profit_dates(stock_prices, dts):
    max_profit = 0
    buy, sell = 0, 0
    for i in range(0, len(stock_prices)-1):
        for j in range(i+1, len(stock_prices)):
            diff = stock_prices[j] - stock_prices[i]
            if max_profit < diff:
                max_profit = diff
                buy = i
                sell = j
    return max_profit, dts[buy], dts[sell]


print(get_profit_dates(stock_prices, dts))

