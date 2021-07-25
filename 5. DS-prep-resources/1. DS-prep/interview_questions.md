### 1. This problem was asked by Airbnb.
### What are some factors that might make testing metrics on the Airbnb platform difficult?

External factors such as the day of week, the time of year, the weather (especially in the case of a travel company like Airbnb), or whether they learned about the website through an online ad or found the site organically. If the testing metrics is focused on the booking aspect, factors such as available inventory and responsiveness of hosts could also make the testing difficult as these factors cannot be controlled by AirBnb.

I guess one aspect that may make it difficult is how long should the experiment be run. A week? 2 weeks? We can decide on this using p-value but that isn't always accurate.

### 2. This problem was asked by Robinhood.
### Write a program to calculate correlation (without any libraries except for math) for two lists X and Y.

Usually we'd be able to use Pearson correlation from `scipy.stats` but since we can't, we can just use the mathematical formula, which is:

![image](https://user-images.githubusercontent.com/39492524/118214063-1d268300-b43d-11eb-8b63-4cf255dfb224.png)
```
def find_pearcorr(x, y, x_val, y_val):
  x_mean = np.mean(x)
  y_mean = np.mean(y)
  # divide into 2 parts, top part of equation, and bottom
  corr_numerator = np.sum((x_val - x_mean) * (y_val - y_mean))
  corr_denominator = np.sqrt(np.sum((x_val - x_mean) ^ 2) * np.sum((y_val - y_mean) ^ 2))\
  return corr_numerator / corr_denominator
  ```

### 3. This problem was asked by Lyft.
### How many cards would you expect to draw from a standard deck before seeing the first ace?

We have to look into hypergeometric distribution, which is a discrete probability distribution that describes the probability of k successes (random draws for which the object drawn has a specified feature) in n draws, without replacement, from a finite population of size {\displaystyle N}N that contains exactly K objects with that feature.

Note that this is for without replacement, if there is replacement, then it would be a different case. In this case, we are looking at the following formula:

![image](https://user-images.githubusercontent.com/39492524/118382461-e62ea980-b5c3-11eb-8092-7942316c8ed0.png)

That gives us:

![image](https://user-images.githubusercontent.com/39492524/118382468-f8104c80-b5c3-11eb-81ce-988f51c0b32f.png)

So, the expected number of cards to get the first ace is 53/5

### 4. This problem was asked by Robinhood.
### A and B are playing a game where A has n+1 coins, B has n coins, and they each flip all of their coins. What is the probability that A will have more heads than B?

We can look at it this way, since A has one extra coin, by theory, the chances of getting more tail from A is the same as getting more heads from A, since that extra coin toss could be heads or tails, with a probability of 1/2. Given that A has n+1 coins and B has n coins, A either obtains more heads than B, or more tails than B but never both.

Based on this, we can conclude that the probability that A have more heads is 1/2, same as the probability that A have more tails than B.

### 4. This problem was asked by Facebook.
### Let’s say that you are the first person working on the Facebook News Feed. What metrics would you track and how would you improve those metrics?
Click through rate (CTR), Engagements (Reactions, comments, shares), the demographics of people who engaged with the news feed, reach, impressions, costs per action
