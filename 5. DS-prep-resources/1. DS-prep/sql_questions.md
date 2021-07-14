## 1. This problem was asked by Facebook.

Assume you are given the below tables on users and user posts. Write a query to get the distribution of the number of posts per user.

### users
column_name	type
user_id	    integer
date	      datetime

### posts
column_name	type
post_id	    integer
user_id	    integer
body	      string
date	      datetime

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
