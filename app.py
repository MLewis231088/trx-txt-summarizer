import streamlit as st
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth',-1)
import random
random.seed(0)
import warnings
warnings.filterwarnings('ignore')

import spacy
import re
from text_preprocessing import preprocess_text
from text_preprocessing import * 

import sumy
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

import gensim
from gensim.summarization import summarize
import torch
device = torch.device('cpu')
import transformers
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
# from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)
warnings.filterwarnings("ignore", category=DeprecationWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)


class DataFrame_Loader():    
    def __init__(self):        
        print("Loading the DataFrame file....")
        
    def read_csv(self,data):
        self.df = pd.read_csv(data)
        return self.df


def main():
  st.title("Text Summarization")
  # st.info("Text Summarization")
  
  activities = ["Build Models for Text Summarization"]
  task = st.sidebar.selectbox("Please select your preferred task",activities)

  if task == 'Build Models for Text Summarization':
    model = st.selectbox('Select the model', ('Model 1', 'Model 2', 'Model 3','Model 4'))
    
    # LSA (Latent Semantic Analysis) Model
    if model == "Model 1":
      sen_length_1 = st.slider('Select the sentence length of the summary to be generated', 0, 100, 10)
      with st.form(key='my_form1'):
        text_input1 = st.text_area("Please enter the text to summarize:")
        submit_button1 = st.form_submit_button(label='Summarize the text')
        
      if submit_button1:
        parser=PlaintextParser.from_string(text_input1,Tokenizer('english'))
        # creating the summarizer
        lsa_summarizer=LsaSummarizer()
        lsa_summary= lsa_summarizer(parser.document,sen_length_1)
        # Printing the summary
        st.write('Summary')
        for m1_output in lsa_summary:
          st.write (m1_output)

    # Model 2 - GENSIM - Summarizer by Word Count
    elif model == "Model 2":
      sen_length_2 = st.slider('Select the word count of the summary to be generated', 0, 250, 10)
      with st.form(key='my_form2'):
        text_input2 = st.text_area("Please enter the text to summarize:")
        submit_button2 = st.form_submit_button(label='Summarize the text')
        
      if submit_button2:
        m2_output=summarize(text_input2, word_count=sen_length_2)
        # Printing the summary
        st.write('Summary')
        st.write (m2_output)

    # Model 3 - T5 Model (Abstractive Text Summarization)
    elif model == "Model 3":
      # sen_length_3 = st.slider('Select the word count of the summary to be generated', 0, 250, 10)
      with st.form(key='my_form3'):
        text_input3 = st.text_area("Please enter the text to summarize:")
        submit_button3 = st.form_submit_button(label='Summarize the text')
        
      if submit_button3:
        my_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        # Concatenating the word "summarize:" to raw text
        text = "summarize:" + text_input3
        # encoding the input text
        input_ids=tokenizer.encode(text, return_tensors='pt', max_length=1024).to(device)
        # Generating summary ids
        summary_ids = my_model.generate(input_ids)
        # for ids in summary_ids:
        #   st.write("The summary ids of the generated text are",ids)
        # Printing the summary
        st.write('Summary')
        # Decoding the tensor and printing the summary.
        m3_output = tokenizer.decode(summary_ids[0])
        st.write (m3_output)
        st.write("The number of words in the summarized text are:", len(m3_output.split()))


    # Model 4 - BART Transformers (Abstractive Text Summarization)
    elif model == "Model 4":
      # sen_length_4 = st.slider('Select the word count of the summary to be generated', 0, 250, 10)
      with st.form(key='my_form4'):
        text_input4 = st.text_area("Please enter the text to summarize:")
        submit_button4= st.form_submit_button(label='Summarize the text')
        
      if submit_button4:
        from transformers import pipeline
        from transformers.modeling_bart import BartModel
        from transformers.tokenization_roberta import RobertaTokenizer

        tokens = RobertaTokenizer.from_pretrained('roberta-base').encode(text_input4, return_tensors="pt")
        bart = BartModel.from_pretrained('/kaggle/input/bart-base-huggingface/')
        output = bart(tokens)
#         tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
#         model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
#         # Encoding the inputs and passing them to model.generate()
#         inputs = tokenizer.batch_encode_plus(text_input4, return_tensors='pt')['input_ids'].to(device)         # , max_length=124, padding="max_length", truncation=True
#         summary_ids = model.generate(inputs, early_stopping=True)
#         # Decoding and printing the summary
#         m4_output = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
        st.write (output)
        st.write("The number of words in the summarized text are:", len(m4_output.split()))
      
      
if __name__ == '__main__':
    load = DataFrame_Loader()
    main()
