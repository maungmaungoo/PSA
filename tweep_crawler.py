from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from datetime import datetime
from time import sleep
import time
import os
import json
import re
import nltk
import sqlite3
import pickle
from dotenv import load_dotenv
from sqlite3 import Error
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import spacy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError
from random import randint
from urllib3.exceptions import ProtocolError

# Load environment variables
load_dotenv()
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.getenv('ACCESS_TOKEN_SECRET')
CONSUMER_KEY = os.getenv('CONSUMER_KEY')
CONSUMER_SECRET = os.getenv('CONSUMER_SECRET')

# English words
words = set(nltk.corpus.words.words())

# English stop words
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['rt'])

user_agent = 'user_me_{}'.format(randint(10000,99999))
geolocator = Nominatim(user_agent=user_agent)

nlp = spacy.load('en_core_web_lg')

model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
separator = ', '

POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
SEQUENCE_LENGTH = 300

# Database name
database = r"tweets.db"

# Database table
sql_create_tweets_table = """ CREATE TABLE IF NOT EXISTS tweets (
                                    id integer PRIMARY KEY,
                                    date text NOT NULL,
                                    screen_name text,
                                    tweet text,
                                    tweet_id text,
                                    location text,
                                    latitude real,
                                    longitude real,
                                    hashtags text,
                                    sentiment text
                            ); """

class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        pass

    def stream_tweets(self, hash_tag_list):
        # This handles Twitter authentification and the connection to Twitter Streaming API
        listener = StdOutListener()
        auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords:
        while True:
            try:
                stream.filter(languages=["en"], track = hash_tag_list)
            except ProtocolError:
                print('[!] PROTOCOL ERROR: Connection broken: Retrying...')
                continue
            except Exception as e:
                print('[!] TWEEPY ERROR: %s: Retrying...' % str(e))
                continue

class StdOutListener(StreamListener):
    """
    This is a basic listener that jsut prints received tweets to stdout.
    """
    def __init__(self):
        pass

    def on_data(self, data):
        all_data = json.loads(data)
        hashtags = separator.join([tags['text'] for tags in all_data['entities']['hashtags']])
        tweet = all_data["text"]
        dtime = all_data["created_at"]
        dtime = datetime.strftime(datetime.strptime(dtime,'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d')
        tweet_id = all_data['id']
        screen_name = all_data['user']['screen_name']
        latitude = 0.0
        longitude = 0.0
        location = ''
        if all_data['user']['location']:
            doc = nlp(all_data['user']['location'])
            for ent in doc.ents:
                if ent.label_ == "GPE":
                    try:
                        geoinfo = geolocator.geocode(ent)
                    except GeocoderTimedOut:
                        print('[!] TIMED OUT: GeocoderTimedOut: Retrying...')
                        sleep(randint(1*100,1*100)/100)
                        continue
                    except GeocoderUnavailable:
                        print('[!] UNAVAILABLE: GeocoderUnavailable: Retrying...')
                        sleep(randint(1*100,1*100)/100)
                        continue
                    except GeocoderServiceError:
                        print('[!] NON-SUCCESSFUL: GeocoderServiceError: Retrying...')
                        sleep(randint(1*100,1*100)/100)
                        continue
                    except Exception as e:
                        print('[!] GEOCODER ERROR: %s: Retrying...' % str(e))
                        sleep(randint(1*100,1*100)/100)
                        continue
                    location = str(ent).capitalize()
                    if geoinfo:
                        latitude = geoinfo.latitude
                        longitude = geoinfo.longitude
                        break
        try:
            tweet = re.sub(TEXT_CLEANING_RE, ' ', str(tweet).lower()).strip()
            tokens = []
            for token in tweet.split():
                if token not in stopwords:
                    tokens.append(token)
            tweet = " ".join(tokens)
            time.sleep(0.1)
            sentiment = predict(tweet)
            add_to_database(dtime, screen_name, tweet, tweet_id, location, latitude, longitude, hashtags, sentiment)
            print("[\u2713] A %s tweet added to Database." % str(sentiment))
            return True
        except BaseException as e:
            print("[!] Error on_data: %s" % str(e))
        return True

    def on_exception(self, exception):
        print("[!] Error on_exception %s" % str(exception))
        return False

    def on_error(self, status):
        print("[!] Error on_error status code %d" % int(status))
        return False

# Decode sentiment score into label
def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE
        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predict(text, include_neutral=True):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)
    return label

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def create_tweets(conn, tweet):
    """
    Create a new tweet sentiment into the tweets table
    :param conn:
    :param tweet:
    :return: tweet id
    """
    sql = ''' INSERT INTO tweets(date,screen_name,tweet,tweet_id,location,latitude,longitude,hashtags,sentiment)
              VALUES(?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, tweet)
    conn.commit()
    return cur.lastrowid

def add_to_database(dtime, screen_name, tweet, tweet_id, location, latitude, longitude, hashtags, sentiment):
    conn = create_connection(database)
    with conn:
        # create a new tweet
        data = (str(dtime), str(screen_name), str(tweet), str(tweet_id), str(location), float(latitude), float(longitude), str(hashtags), str(sentiment))
        create_tweets(conn, data)

if __name__ == '__main__':
    conn = create_connection(database)
    with conn:
        # create a new table
        create_table(conn, sql_create_tweets_table)
    hash_tag_list = ["trump"]
    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets(hash_tag_list)