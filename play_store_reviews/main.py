import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adamax

df = pd.read_csv('googleplaystore_user_reviews.csv')

def preprocessing(df):
    df = df.dropna()
    df = df.drop(columns=['App', 'Sentiment_Polarity', 'Sentiment_Subjectivity'])
    category = pd.get_dummies(df.Sentiment)
    df_baru = pd.concat([df, category], axis=1)
    df_baru = df_baru.drop(columns='Sentiment')
    df_baru
    return df_baru

preprocessing(df)

ulasan = df_baru['Translated_Review'].values
label = df_baru[['Negative', 'Neutral', 'Positive']].values
ulasan_latih, ulasan_test, label_latih, label_test = train_test_split(ulasan, label, test_size=0.2)

tokenizer = Tokenizer(num_words=5000, oov_token='x')
tokenizer.fit_on_texts(ulasan_latih) 
tokenizer.fit_on_texts(ulasan_test)

sekuens_latih = tokenizer.texts_to_sequences(ulasan_latih)
sekuens_test = tokenizer.texts_to_sequences(ulasan_test)
 
padded_latih = pad_sequences(sekuens_latih) 
padded_test = pad_sequences(sekuens_test)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_acc') > 0.92):
          print('\nval_accuracy telah mencapai lebih dari 92%.')
          self.model.stop_training = True
callbacks = myCallback()

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='binary_crossentropy', optimizer=Adamax(learning_rate=0.001), metrics=['accuracy'])

history = model.fit(padded_latih, label_latih, epochs=30, 
                    validation_data=(padded_test, label_test), verbose=2, steps_per_epoch=250, callbacks=[callbacks])