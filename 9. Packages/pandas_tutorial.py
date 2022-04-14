import pandas as pd
import numpy as np


# For the first part, let's focus on string manipulation in a Pandas dataframe'
df = pd.Series(['Gulshan', 'Shashank', 'Bablu',
                'Abhishek', 'Anand', np.nan, 'Pratap'])

print(df)

# We cam see that now its an object, we can convert dataframe to string using ".astype"
print(df.dtype)

df = df.astype('string')
print(df.dtype)

# Let's try a different dataframe
df = pd.Series(['night_fury1', 'Is  ', 'Geeks, forgeeks',
                '100', np.nan, '  Contributor '], dtype='string')

# convert all uppercase to lowercase letters
print(df.str.lower())

# and the reverse
print(df.str.upper())

# Let's move on to regex in Pandas
import re

# Let's create a Dataframe
df = pd.DataFrame({'City': ['New York (City)', 'Prague', 'New Delhi (Delhi)', 'Venice', 'new Orleans'],
                   'Date': ['13/10/2015 12:30:00.145', '13/11/2015 00:33:00.972', '13/10/2010 00:30:00.642',
                            '13/10/2009 11:30:00.325', '13/10/2015 12:12:00.321'],
                   'Cost': [10000, 5000, 15000, 2000, 12000]})

# to search for opening brackets in the name
match1 = re.search('\(.*?\)', 'New York (City)')
match2 = re.search('(\d{2,}?\s+to\s+\d{2,}?)', 'from 121 to 50')

df['new_Date'] = pd.to_datetime(df.Date, format='%d/%m/%Y %H:%M:%S.%f')

# If you want to filter by only seeing the ones with City name starting with 'P'
df_extract = df[df['City'].str.match(r'(^P.*)') == True]

