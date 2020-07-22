import json
from nltk_utils import bag_of_words, tokenize
import numpy as np
import random
import torch
import torch.nn as nn
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader
import streamlit as st


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)
    
FILE = "data.pth"
data = torch.load(FILE)


 
input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']


model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()




bot_name = "Sam"
st.write(f"{bot_name}: Let's Chat! Type 'quit' to exit")



def chatbot(sentence):

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
        
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                
                st.write(f"{bot_name}: {random.choice(intent['responses'])}")
                    
                
                    
    else:
        st.write(f"{bot_name}: I do not understand...")
    


def talk():
    
    sentence = st.text_input("You: ", key = 1)
        
    if st.button('Submit'):
            
        if sentence == "quit":
            st.write(f"{bot_name}: Goodbye, feel free to ask any questions")
        else:
            chatbot(sentence)
             
    
talk()


        
    


        
        
               

    
        
