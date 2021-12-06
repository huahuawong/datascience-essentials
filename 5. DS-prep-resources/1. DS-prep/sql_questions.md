## 1. This problem was asked by Facebook.

Assume you are given the below tables on users and user posts. Write a query to get the distribution of the number of posts per user.

### users
column_name	type\
user_id	integer\
date	       datetime

### posts
column_name	    type\
post_id	    integer\
user_id	    integer\
body	           string\
date	           datetime

- Note: I am not sure what does the question mean by 'distribution', but this is how I'd do for frequency distribution
```
SELECT COUNT(p.post_id) FROM users u JOIN posts p
ON u.user_id = p.user_id
GROUP BY u.user_id
```

If we want to see the number of posts in terms of range/ buckets. Alternatively, we can use `width_bucket`

```
WITH T1 AS(
SELECT COUNT(p.post_id) AS count_post FROM users u JOIN posts p
ON u.user_id = p.user_id
GROUP BY u.user_id)

SELECT width_bucket(count_post, 0, 50, 5) as bucket, 
       count(*) as cnt FROM T1
group by bucket 
order by bucket;
```

This will give us bucket1, 2, 3, 4 and 5 (0-10, 11-20, etc.) and shows the count of users that falls into each specific range of number of posts.


## 2. This problem was asked by Opendoor.

Assume you are given the below table on house prices from various zip codes that have been listed. Write a query to get the top 5 zip codes by market share of house prices for any zip code with at least 10000 houses.

### house_listings
column_name	type\
house_id	integer\
zip_code	integer\
price	float\
listing_date	datetime

```
WITH t1 AS(
SELECT zip_code, COUNT(*) AS num_house FROM house_listings
GROUP BY zip_code
HAVING COUNT(*) > 10000)

SELECT * FROM house_listings 
WHERE zip_code IN (SELECT zip_code FROM t1)
ORDER BY zip_code ASC
LIMIT 5
```


### 3. This problem was asked by Etsy.

Assume you are given the below table on transactions from users for purchases. Write a query to get the list of customers where their earliest purchase was at least $50.

### user_transactions
column_name	       type\
transaction_id	integer\
product_id	       integer\
user_id	       integer\
spend	              float\
transaction_date	datetime

```
SELECT user_id FROM 
(SELECT user_id, MIN(transaction_date), spend FROM user_transactions GROUP BY user_id)
WHERE spend >= 50
```

