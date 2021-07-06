# Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('ggplot')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# for displaying 500 results in pandas dataframe
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import classification_report,confusion_matrix
from collections import defaultdict,Counter

nltk.download('stopwords')

stop = set(stopwords.words('english'))
plt.style.use('seaborn')

import plotly.offline as py
py.init_notebook_mode(connected=True)

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

# We can see that there are 20% missing values in the 'Location' column, missing data as white lines, however, they will
# not be removed as we are not interested in location right now, we are looking at text data specifically
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
labels = class_df.sentiment

colors = ['#17C37B','#F92969','#FACA0C']

my_pie,_,_ = plt.pie(percent_class,radius = 1.2,labels=labels,colors=colors,autopct="%.1f%%")
plt.setp(my_pie, width=0.6, edgecolor='white')
plt.show()

fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

tweet_len=train[train['sentiment']=="positive"]['text'].str.len()
ax1.hist(tweet_len,color='#17C37B')
ax1.set_title('Positive Sentiments')

tweet_len=train[train['sentiment']=="negative"]['text'].str.len()
ax2.hist(tweet_len,color='#F92969')
ax2.set_title('Negative Sentiments')

tweet_len=train[train['sentiment']=="neutral"]['text'].str.len()
ax3.hist(tweet_len,color='#FACA0C')
ax3.set_title('Neutral Sentiments')

fig.suptitle('Characters in tweets')
plt.show()
# Seems like the distribution is slightly left skewed, but it's nothing crazy

# Text cleaning
# 1. Loading the stop words that will be filtered out from the text
# 2. Clean the data by remving the urls, mentions, hashtags and digits from the text. This can be done using regex


# load stop words
stop_word = stopwords.words('english')


def clean(text):
    # remove urls
    text = re.sub(r'http\S+', " ", text)
    # remove mentions
    text = re.sub(r'@\w+', ' ', text)
    # remove hashtags
    text = re.sub(r'#\w+', ' ', text)
    # remove digits
    text = re.sub(r'\d+', ' ', text)
    # remove html tags
    text = re.sub('r<.*?>', ' ', text)
    # remove stop words
    text = text.split()
    text = " ".join([word for word in text if not word in stop_word])
    return text


train['OriginalTweet'] = train['OriginalTweet'].apply(lambda x: clean(x))
test['OriginalTweet'] = test['OriginalTweet'].apply(lambda x: clean(x))
train.head()

# Notice how the first row's OriginalTweet is empty now? That's because the entire tweet was filled with mentions, and
# stopwords, so the empty rows will now be removed from the dataset
train_df = train[train['OriginalTweet'] != '']
test_df = test[test['OriginalTweet'] != '']

train.shape
train_df.shape
# 15 rows were empty on the ['OriginalTweet'] column

df_train = train_df[['OriginalTweet', 'sentiment']]; df_test = test_df[['OriginalTweet', 'sentiment']]

x_train = df_train['OriginalTweet']; y_train = df_train['sentiment']
x_test = df_test['OriginalTweet']; y_test = df_test['sentiment']

max_len = np.max(x_train.apply(lambda x:len(x)))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
vocab_length = len(tokenizer.word_index) + 1

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

print(f"Vocab length: {vocab_length} and the max sequence length: {max_len}")
embedding_dim = 16


def create_corpus(target):
    corpus = []
    for x in train[train['sentiment'] == target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


corpus = create_corpus("positive")      # positive/ neutral/ negative
counter = Counter(corpus)
most = counter.most_common()
x = []; y = []

for word,count in most[:40]:
    if word not in stop:
        x.append(word)
        y.append(count)
sns.barplot(x=y, y=x)

# Data modeling
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_length, embedding_dim, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3, activation='softmax')   # since we have 3 output classes to be considered
])

# opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model.summary()

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# encode the sentiment to integer values
from keras.utils import to_categorical

clean_up = {"sentiment": {"negative": 0, "neutral": 1, "positive": 2}}
y_train = pd.DataFrame(y_train); y_test = pd.DataFrame(y_test)

y_train = y_train.replace(clean_up)
y_test = y_test.replace(clean_up)

y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

num_epochs = 10
history = model.fit(x_train, y_train, epochs=num_epochs,
                    validation_data=(x_test, y_test))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"Accuracy on training data is:- {acc[-1]*100} %")
print(f"Loss {loss[-1]*100}")

print(f"Accuracy on validation data is:- {val_acc[-1]*100} %")
print(f"Loss {val_loss[-1]*100}")

epochs = range(len(acc))

plt.plot(epochs, acc,'b',label='training acc')
plt.plot(epochs, val_acc, 'r', label='validation acc')
plt.legend()
plt.show()


plt.plot(epochs, loss,'b',label='training loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.legend()
plt.show()

pred = model.predict_classes(x_test)

# Confusion matrix
cm = confusion_matrix(np.argmax(y_test,1),pred)
cm
