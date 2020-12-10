import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, Adamax, SGD
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
sns.set()

data = pd.read_csv('data.csv', encoding='latin-1')
alay_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1')
id_stopword_dict = pd.read_csv('stopwordbahasa.csv', header=None)
alay_dict = alay_dict.rename(columns={'anakjakartaasikasik':'tidak baku', 'anak jakarta asyik asyik':'baku'})
id_stopword_dict = id_stopword_dict.rename(columns={0:'stopword'})

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n',' ',text) # Remove every '\n'
    text = re.sub('rt',' ',text) # Remove every retweet symbol
    text = re.sub('user',' ',text) # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
    text = re.sub('  +', ' ', text) # Remove extra spaces
    return text
    
def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 
    return text

alay_dict_map = dict(zip(alay_dict['tidak baku'], alay_dict['baku']))
def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

def remove_stopword(text):
    text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text) # Remove extra spaces
    text = text.strip()
    return text

def stemming(text):
    return stemmer.stem(text)

def preprocess(text):
    text = lowercase(text)
    text = remove_nonaplhanumeric(text)
    text = remove_unnecessary_char(text)
    text = normalize_alay(text)
    text = stemming(text)
    text = remove_stopword(text)
    return text

data['Tweet'] = data['Tweet'].apply(preprocess)

df = data[['Tweet', 'HS']]
df = df.dropna()

tweet = df['Tweet'].values
label = df[['HS']].values

category = pd.get_dummies(df['HS'])
df_baru = pd.concat([df, category], axis=1)
df_baru = df_baru.drop(columns='HS')
df_baru = df_baru.rename(columns={0:'Not hate speech', 1:'Hate speech'})

tweet_train, tweet_test, label_train, label_test = train_test_split(tweet, label, test_size=0.2)

tokenizer = Tokenizer(num_words=5000, oov_token='x')
tokenizer.fit_on_texts(tweet_train) 

sekuens_latih = tokenizer.texts_to_sequences(tweet_train)
sekuens_test = tokenizer.texts_to_sequences(tweet_test)
 
padded_latih = pad_sequences(sekuens_latih) 
padded_test = pad_sequences(sekuens_test)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.85):
          self.model.stop_training = True
callbacks = myCallback()

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adamax(learning_rate=0.001), metrics=['accuracy'])

history = model.fit(padded_latih, label_train, epochs=100, 
                    validation_data=(padded_test, label_test), verbose=1, callbacks=[callbacks])