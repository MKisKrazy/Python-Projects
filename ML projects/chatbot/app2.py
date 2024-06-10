from flask import Flask, request, jsonify
import os
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
from urllib.parse import unquote  # Import URL decoding function
from googletrans import Translator  # Import Translator from googletrans module

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Open and read the dataset file
f = open('datatamil.txt', 'r', errors='ignore')
raw = f.read().lower()

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("வணக்கம்", "ஹலோ", "வணக்கம்", "வாங்க", "ஏன் தான்", "வணக்கம்", "ஹலோ")
GREETING_RESPONSES = ["வணக்கம்", "ஹலோ", "", "வணக்கம்", "ஹலோ", "நான் சந்திக்க மிக்க மகிழ்ச்சியாக உள்ளேன்"]

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
    # Get the encoded English input text from the URL parameter
    encoded_input = request.args.get('input')

    # Initialize the translator
    translator = Translator()

    # Decode the input text to obtain the original English text
    english_input = unquote(encoded_input)

    # Translate the English input to Tamil
    tamil_input = translator.translate(english_input, src='en', dest='ta').text
    print("Translated Tamil input:", tamil_input)  # Debugging print statement

    # Convert the translated Tamil input to lowercase
    user_response = tamil_input.lower()

    print("Processed user response:", user_response)  # Debugging print statement

    if user_response != 'bye':
        if user_response == 'நன்றி' or user_response == 'நன்றி':
            response_text = "ஆனேகா: நன்றி!"
        else:
            greeting_res = greeting(user_response)
            if greeting_res:
                response_text = "ஆனேகா: " + greeting_res
            else:
                response_text = response(user_response)
                sent_tokens.remove(user_response) if response_text else None
    else:
        response_text = "ஆனேகா: போய்! நல்ல நேரம் கழிக்கைக்கு வேண்டுகிறேன்!"

    return jsonify({'response': response_text})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
