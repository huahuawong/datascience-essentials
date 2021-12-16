### Let’s say you’re given a dataframe of standardized test scores from high schoolers from grades 9 to 12 called df_grades.
### Given the dataset, write code function in Pandas called bucket_test_scores to return the cumulative percentage of students that received scores within the buckets of <50, <75, <90, <100.

Key here is to use pd.cut, it can be used to bin in values into discrete intervals. And then we can use 'groupby' to get the counts at each interval at each grade

```
import pandas as pd

# Initialize cha
data = {'user_id':  [1,2,3,4,5],
        'grade': [10,10,11,10,11],
        'test_score': [85,60,90,30,99]
        }

df = pd.DataFrame(data,columns = ['user_id' ,  'grade' , 'test_score'] )

# Define the bins
bins = [0, 50, 75, 90, 100]
labels=['<50','<75','<90' , '<100']

df['test_score'] = pd.cut(df['test_score'], bins,labels=labels)

numer = df.groupby(['grade','test_score'])['user_id'].count()
denom = df.groupby(['grade'])['user_id'].count()

df = numer/denom
df = df.reset_index()

df['Percentage']=(round(df.groupby(['grade'])['user_id'].cumsum()*100, 0)).astype('str')+"%"

df=df[['grade','test_score','Percentage']]
```
