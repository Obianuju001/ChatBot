import nltk
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()
# Download required NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


data = pd.read_csv('Mental_Health_FAQ.csv')
data.drop('Question_ID', axis = 1, inplace = True)
# data

# Here, the Next step is tokenize our text dataset.<br>
# There are two types of tokenization:
#     <ol><li>Word Tokenization: This is  the process of breaking down a text or document into individual words or tokens.</li>
#     <li>Sent Tokenization: This is to break down the text data into individual sentences so that each sentence can be processed separately.</li><br></ol>
# Lemmatization: The goal of lemmatization is to reduce a word to its canonical form so that variations of the same word can be treated as the same token<br>
# For example, the word "jumped" may be lemmatized to "jump", and the word "walking" may be lemmatized to "walk".<br>
# By reducing words to their base forms, lemmatization can help to simplify text data and reduce the number of unique tokens that need to be analyzed or processed.


# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    global tokens
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


data['tokenized Questions'] = data['Questions'].apply(preprocess_text)
# data.head()


corpus = data['tokenized Questions'].to_list()
# corpus


tfidf_vector = TfidfVectorizer()
v_corpus = tfidf_vector.fit_transform(corpus) 
# print(v_corpus)



# ----------------------------------------- STREAMLIT DESIGN -----------------------------------------
st.markdown("<h1 style = 'color: #4A55A2; text-align: center; font-family: montserrat '>MENTAL HEALTH CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -25px; color: #7895CB; text-align: center; font-family: script'>Built By Obianuju</h4>", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)
st.markdown("<h4 style = 'margin: -25px; color: #7895CB; text-align: center; font-family: script'>Welcome To Orpheus ChatBox</h4>", unsafe_allow_html = True)


# st.markdown("<br> <br>", unsafe_allow_html= True)
col1, col2 = st.columns(2)
col1.image('pngwing.com (7).png', caption = 'Mental Health Related Chats')


# Putting it all in a function

def bot_response(user_input):
    user_input_processed = preprocess_text(user_input)
    v_input = tfidf_vector.transform([user_input_processed])
    most_similar = cosine_similarity(v_input, v_corpus)
    most_similar_index = most_similar.argmax()
    
    return data['Answers'].iloc[most_similar_index]


import random

chatbot_greeting = [
    "Hello there, Welcome to Orpheus Bot. Please enjoy your usage",
    "Hi user, This bot is created by Orpheus, enjoy your usage",
    "Hi hi, How you dey my nigga",
    "Alaye mi, Abeg enjoy your usage",
    "Hey Hey, pls enjoy your usage"
]
user_greeting = ["hi", "hello there", "hello", "hey", "hi there"]
exit_word = ["bye", "thanks bye", "thanks", "exit", "goodbye"]

# # print(f'\t\t\t\tWelcome To Orpheus ChatBox\n\n')
# while True:
#     user_q = input('Pls ask your mental illness related question: ')
#     if user_q in user_greeting:
#         print(random.choice(chatbot_greeting))
#     elif user_q in exit_word:
#         print('Thank you for your usage. Bye')
#         break
#     else:
#         responses = bot_response(user_q)
#         print(f'Question: {user_q}')
#         print(f'ChatBot: {responses}')



# st.write(f'\t\t\t\tWelcome To Orpheus ChatBox\n\n')
# while True:
user_q = col2.text_input('Pls ask your mental illness related question: ')
if user_q in user_greeting:
        col2.write(random.choice(chatbot_greeting))
elif user_q in exit_word:
        col2.write('Thank you for your usage. Bye')
        # break
elif user_q == '':
        st.write('')
else:
        responses = bot_response(user_q)
        col2.write(f'Question: {user_q}')
        col2.write(f'ChatBot: {responses}')

