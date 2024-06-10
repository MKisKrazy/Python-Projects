from flask import Flask, request, jsonify
import os
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
from googletrans import Translator  # Import Translator from googletrans module

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Open and read the dataset file
f = open('datasset.txt', 'r', errors='ignore')
raw = f.read().lower()

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey","hii")
GREETING_RESPONSES = ["hi", "hey", "", "hi there", "hello", "I am glad! you are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=["என்", "உடன்", "அதன்", "இதன்"])
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        chatbot_response = "மன்னிக்கவும்! நான் உங்களை படிக்க முடியவில்லை"
        return chatbot_response
    else:
        chatbot_response = sent_tokens[idx]
        return chatbot_response

@app.route('/chatbot', methods=['GET'])
def chatbot_api():
    # Get the Tamil input text from the URL parameter
    tamil_input = request.args.get('input')

    # Translate the Tamil input to English
    translator = Translator()
    english_input = translator.translate(tamil_input, src='ta', dest='en').text

    # Process the translated English input
    user_response = english_input.lower()

    if user_response != 'bye':
        if user_response == 'thank you' or user_response == 'thanks':
            response_text = " Thank you!"
        else:
            greeting_res = greeting(user_response)
            if greeting_res:
                response_text =  greeting_res
            else:
                response_text = response(user_response)
                sent_tokens.remove(user_response) if response_text else None
    else:
        response_text = "Goodbye! Have a nice day!"

    # Translate the English response back to Tamil
    translated_response = translator.translate(response_text, src='en', dest='ta').text

    return jsonify({'response': translated_response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)