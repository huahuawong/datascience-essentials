## These are all in PostgreSQL

### 1. You are given a table named repositories, format as below:

** repositories table schema **

project
commits
contributors
address
The table shows project names of major cryptocurrencies, their numbers of commits and contributors and also a random donation address ( not linked in any way :) ).

Your job is to remove all numbers in the address column and replace with '!', then return a table in the following format:

** output table schema **

project
commits
contributors
address
Case should be maintained.

Answer:
`SELECT project, commits, contributors, regexp_replace(address,'[0-9]','!', 'g' ) AS address FROM repositories;`

Alternatively, we can use '[[:digit:]]' or '/d' in place of '[0-9]'. 
'g' is to make sure in case there are multiple numbers that needs to be replaced.

### 2. Given a demographics table in the following format:

** demographics table schema **

id
name
birthday
race
you need to return a table that shows a count of each race represented, ordered by the count in descending order as:

** output table schema **

race
count

Answer: 
`SELECT race, COUNT(race)
FROM demographics
GROUP BY race
ORDER BY Count(race) desc`

### 3. Given a demographics table in the following format:

** demographics table schema **

id
name
birthday
race
return a single column named 'calculation' where the value is the bit length of name, added to the number of characters in race.

Answer:
`SELECT (BIT_LENGTH(name) + LENGTH(race)) AS calculation FROM demographics;`

Alternatively, CHAR_LENGTH can be used in place of LENGTH

