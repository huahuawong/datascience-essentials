### 1. This problem was asked by Airbnb.
### What are some factors that might make testing metrics on the Airbnb platform difficult?

External factors such as the day of week, the time of year, the weather (especially in the case of a travel company like Airbnb), or whether they learned about the website through an online ad or found the site organically. If the testing metrics is focused on the booking aspect, factors such as available inventory and responsiveness of hosts could also make the testing difficult as these factors cannot be controlled by AirBnb.

I guess one aspect that may make it difficult is how long should the experiment be run. A week? 2 weeks? We can decide on this using p-value but that isn't always accurate.

### 2. This problem was asked by Robinhood.
### Write a program to calculate correlation (without any libraries except for math) for two lists X and Y.

Usually we'd be able to use Pearson correlation from `scipy.stats` but since we can't, we can just use the mathematical formula, which is:
![image](https://user-images.githubusercontent.com/39492524/118214063-1d268300-b43d-11eb-8b63-4cf255dfb224.png)
`
def find_pearcorr(x, y, x_val, y_val):
  x_mean = np.mean(x)
  y_mean = np.mean(y)
  # divide into 2 parts, top part of equation, and bottom
  corr_numerator = np.sum((x_val - x_mean) * (y_val - y_mean))
  corr_denominator = np.sqrt(np.sum((x_val - x_mean) ^ 2) * np.sum((y_val - y_mean) ^ 2))\
  return corr_numerator / corr_denominator
`  
