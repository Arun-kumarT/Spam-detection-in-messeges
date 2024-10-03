import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import pickle
from nltk.stem import WordNetLemmatizer
import re
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
nltk.download('wordnet')
st.title("Spam Detection")

model1 = pickle.load(open('model.pkl', 'rb'))

class prediction:
    
    def __init__(self,data):
        self.data = data
        
    def user_data_preprocessing(self):
        lm = WordNetLemmatizer()
        review = re.sub('^a-zA-Z0-9',' ',self.data)
        review = review.lower()
        review = review.split()
        review = [data for data in review if data not in stopwords.words('english')]
        review = [lm.lemmatize(data) for data in review]
        review = " ".join(review)
        return [review]
    
    def user_data_prediction(self):
        preprocess_data = self.user_data_preprocessing()
        
        if model1.predict(preprocess_data)[0] == 'spam':
            return 'This Message is Spam'
            
        else:
            return 'This Message is Legit' 


text=st.text_input("Enter a message to predict spam or not")

if st.button("Prdict"):
    result=prediction(text).user_data_prediction()
    st.write(result)
