# Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
plt.style.use('ggplot')

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

#for displaying 500 results in pandas dataframe
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


import re
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict,Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import string
nltk.download('stopwords')

stop = set(stopwords.words('english'))
plt.style.use('seaborn')

from plotly import tools
import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import textstat
from textblob import TextBlob
from tqdm import tqdm
from statistics import *
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Inspecting the dataset
train = pd.read_csv("../nlp_data/Corona_NLP_train.csv",encoding='latin1')
test = pd.read_csv("../nlp_data/Corona_NLP_test.csv",encoding='latin1')

df = pd.concat([train,test])
df.head()
df.shape
df.info()

# The 'Original tweet' and 'Sentiment' is currently of object type, they will be converted to strings
df['OriginalTweet'] = df['OriginalTweet'].astype(str)
df['Sentiment']=df['Sentiment'].astype(str)

for dataset in [train, test]:
    dataset['OriginalTweet'] = dataset['OriginalTweet'].astype(str)
    dataset['Sentiment'] = dataset['Sentiment'].astype(str)

# Check for null values
null = df.isnull().sum().sort_values(ascending=False)
total = df.shape[0]
percent_missing = (df.isnull().sum() / total).sort_values(ascending=False)

missing_data = pd.concat([null, percent_missing], axis=1, keys=['Total missing', 'Percent missing'])

missing_data.reset_index(inplace=True)
missing_data = missing_data.rename(columns={"index": " column name"})

print("Null Values in each column:\n", missing_data)

# We can see that there are 20% missing values in the 'Location' column, missing data as white lines
import missingno as msno
msno.matrix(df,color=(0.3,0.36,0.44))
plt.show()

print('Total tweets in this data: {}'.format(df.shape[0]))
print('Total Unique Users in this data: {}'.format(df['UserName'].nunique()))       # Shows no duplicate users

print(df.Sentiment.value_counts())

# We will copy the text in another column so that the original text is also there for comparison
df['text'] = df.OriginalTweet
df["text"] = df["text"].astype(str)

for dataset in [train, test]:
    dataset['text'] = dataset.OriginalTweet
    dataset["text"] = dataset["text"].astype(str)


# Data has 5 classes, let's convert them to 3
def classes_def(x):
    if x == "Extremely Positive":
        return "positive"
    elif x == "Extremely Negative":
        return "negative"
    elif x == "Negative":
        return "negative"
    elif x == "Positive":
        return "positive"
    else:
        return "neutral"


df['sentiment'] = df['Sentiment'].apply(lambda x: classes_def(x))
train['sentiment'] = train['Sentiment'].apply(lambda x: classes_def(x))
test['sentiment'] = test['Sentiment'].apply(lambda x: classes_def(x))
target = df['sentiment']

df.sentiment.value_counts(normalize=True)

# Checking class distribution
class_df = df.groupby('sentiment').count()['OriginalTweet'].reset_index().sort_values(by='OriginalTweet', ascending=False)
class_df.style.background_gradient(cmap='winter')

percent_class = class_df.OriginalTweet
labels  class_df.sentiment

colors = ['#17C37B','#F92969','#FACA0C']

my_pie,_,_ = plt.pie(percent_class,radius = 1.2,labels=labels,colors=colors,autopct="%.1f%%")
plt.setp(my_pie, width=0.6, edgecolor='white')
plt.show()

# fig=make_subplots(1,2,subplot_titles=('Train set','Test set'))
# x=train.sentiment.value_counts()
# fig.add_trace(go.Bar(x=x.index,y=x.values,marker_color=['#17C37B','#F92969','#FACA0C'],name='train'),row=1,col=1)
# x=test.sentiment.value_counts()
# fig.add_trace(go.Bar(x=x.index,y=x.values,marker_color=['#17C37B','#F92969','#FACA0C'],name='test'),row=1,col=2)
# plt.show()

