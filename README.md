# MINI-PROJECT
#Action recommendation model for maximization of profit in cryptocurrencies based on Twitter sentiments analysis

                                          SOURCE CODE
**EXTRACTING TEXT:**

print("READING CSV FILE FROM DIRECTORY")
df = pd.read_csv(r'/content/tweets_30000_1.csv')
print(df.head())
print(' ')


PRE-PROCESSING THE TEXT

import nltk
from nltk.stem.wordnet import WordNetLemmatizer


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = nltk.corpus.stopwords.words(['english'])




print(stop_words)
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()


def cleaning(data):
  #remove urls
  tweet_without_url = re.sub(r'http\S+',' ', data)


  #remove hashtags
  tweet_without_hashtag = re.sub(r'#\w+', ' ', tweet_without_url)


  #3. Remove mentions and characters that not in the English alphabets
  tweet_without_mentions = re.sub(r'@\w+',' ', tweet_without_hashtag)
  precleaned_tweet = re.sub('[^A-Za-z]+', ' ', tweet_without_mentions)


    #2. Tokenize
  tweet_tokens = TweetTokenizer().tokenize(precleaned_tweet)


    #3. Remove Puncs
  tokens_without_punc = [w for w in tweet_tokens if w.isalpha()]


    #4. Removing Stopwords
  tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]


    #5. lemma
  text_cleaned = [lem.lemmatize(t) for t in tokens_without_sw]


    #6. Joining
  return " ".join(text_cleaned)
def getSubjectivity(tweet):
  return TextBlob(tweet).sentiment.subjectivity


def getPolarity(tweet):
  return TextBlob(tweet).sentiment.polarity
df['subjectivity'] = df['cleaned_text'].apply(getSubjectivity)
df['polarity'] = df['cleaned_text'].apply(getPolarity)
df.head()



IMPLEMENTING  ALGORITHM
(1)LSTM IN TEXT
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
batch_size = 32
epochs = 5
max_features = 20000
embed_dim = 100
np.random.seed(seed)
K.clear_session()
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
(2)GRU IN TEXT
 import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
batch_size = 32
epochs = 5
max_features = 20000
embed_dim = 100
np.random.seed(seed)
K.clear_session()
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
(3)LSTM IN COIN
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import Callback
import numpy as np
# Your data loading and preprocessing code goes here...
# Define a threshold to determine if the prediction is correct (e.g., within a certain range)
threshold = 0.02  # You can adjust this threshold as needed


# Define a custom callback to calculate and print accuracy
class CustomAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(xtest)
        accuracy = np.mean(np.abs(ytest - y_pred) < threshold)
        print(f'\nEpoch {epoch+1}/{self.params["epochs"]} - Accuracy: {accuracy * 100:.2f}%\n')

# Define your LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# Compile the model with MSE as the loss
model.compile(optimizer='adam', loss='mean_squared_error')


# Create the custom callback instance
accuracy_callback = CustomAccuracyCallback()


# Train the model with the custom accuracy callback
model.fit(xtrain, ytrain, batch_size=1, epochs=5, validation_data=(xtest, ytest), callbacks=[accuracy_callback])



(4) GRU IN COIN
from keras.models import Sequential
from keras.layers import Dense, GRU
model = Sequential()
model.add(GRU(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(GRU(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

(5) CLASSIFICATION CODE FOR COIN 	
 from keras.metrics import MeanAbsoluteError
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanAbsoluteError()])
history = model.fit(xtrain, ytrain, batch_size=1, epochs=30)


# Access and print the loss and MAE values
loss_values = history.history['loss']
mae_values = history.history['mean_absolute_error']


for epoch, (loss, mae) in enumerate(zip(loss_values, mae_values), 1):
    print(f"Epoch {epoch}/{len(loss_values)} - Loss: {loss:.4f} - MAE: {mae:.4f}")

