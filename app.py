import streamlit as st
import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', -1)
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
from transformers import BartTokenizer, BartForConditionalGeneration

PAGE_CONFIG = {"page_title": "StColab.io", "page_icon": ":smiley:", "layout": "centered"}
st.set_page_config(**PAGE_CONFIG)
warnings.filterwarnings("ignore", category=DeprecationWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)


class DataFrame_Loader():
    def __init__(self):
        print("Loading the DataFrame file....")

    def read_csv(self, data):
        self.df = pd.read_csv(data)
        return self.df


def main():
    st.title("Text Summarization")
    # st.info("Text Summarization")

    activities = ["Models for Text Summarization"]
    task = st.sidebar.selectbox("Please select your preferred task", activities)

    if task == 'Models for Text Summarization':
        model = st.selectbox('Select the model', ('Model 1', 'Model 2', 'Model 3', 'Model 4'))

        # LSA (Latent Semantic Analysis) Model
        if model == "Model 1":
            sen_length_1 = st.slider('Select the sentence length of the summary to be generated', 0, 100, 10)
            with st.form(key='my_form1'):
                text_input1 = st.text_area("Please enter the text to summarize:")
                submit_button1 = st.form_submit_button(label='Summarize the text')

            if submit_button1:
                parser = PlaintextParser.from_string(text_input1, Tokenizer('english'))
                # creating the summarizer
                lsa_summarizer = LsaSummarizer()
                lsa_summary = lsa_summarizer(parser.document, sen_length_1)
                # Printing the summary
                st.write('Summary')
                for m1_output in lsa_summary:
                    st.success(m1_output)

        # Model 2 - GENSIM - Summarizer by Word Count
        elif model == "Model 2":
            sen_length_2 = st.slider('Select the word count of the summary to be generated', 0, 250, 10)
            with st.form(key='my_form2'):
                text_input2 = st.text_area("Please enter the text to summarize:")
                submit_button2 = st.form_submit_button(label='Summarize the text')

            if submit_button2:
                m2_output = summarize(text_input2, word_count=sen_length_2)
                # Printing the summary
                st.write('Summary')
                st.success(m2_output)

        # Model 3 - T5 Model (Abstractive Text Summarization)
        elif model == "Model 3":
            _num_beams = 4
            _no_repeat_ngram_size = 3
            _length_penalty = 2
            _min_length = 30
            _max_length = 200
            _early_stopping = True

            col1, col2 = st.beta_columns(2)
            _min_length = col1.number_input("min_length", value=_min_length)
            _max_length = col2.number_input("max_length", value=_max_length)
            # _num_beams = col1.number_input("num_beams", value=_num_beams)
            # _no_repeat_ngram_size = col2.number_input("no_repeat_ngram_size", value=_no_repeat_ngram_size)
            # _length_penalty = col3.number_input("length_penalty", value=_length_penalty)
            # _early_stopping = col3.number_input("early_stopping", value=_early_stopping)


            with st.form(key='my_form3'):
                text_input3 = st.text_area("Please enter the text to summarize:")
                submit_button3 = st.form_submit_button(label='Summarize the text')

            if submit_button3:
                my_model = T5ForConditionalGeneration.from_pretrained('t5-small')
                tokenizer = T5Tokenizer.from_pretrained('t5-small')
                # Concatenating the word "summarize:" to raw text
                text = "summarize:" + text_input3
                # encoding the input text
                input_ids = tokenizer.encode(text, return_tensors='pt', max_length=1024).to(device)
                # Generating summary ids
                summary_ids = my_model.generate(input_ids,
                                                num_beams=_num_beams,
                                                no_repeat_ngram_size=_no_repeat_ngram_size,
                                                length_penalty=_length_penalty,
                                                min_length=_min_length,
                                                max_length=_max_length,
                                                early_stopping=_early_stopping
                                                )

                st.write('Summary')
                # Decoding the tensor and printing the summary.
                m3_output = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                st.success(m3_output)
                st.write("The number of words in the summarized text are:", len(m3_output.split()))


        # Model 4 - BART Transformers (Abstractive Text Summarization)
        elif model == "Model 4":
            _num_beams = 4
            _no_repeat_ngram_size = 3
            _length_penalty = 1
            _min_length = 12
            _max_length = 128
            _early_stopping = True

            col1, col2 = st.beta_columns(2)
            _min_length = col1.number_input("min_length", value=_min_length)
            _max_length = col2.number_input("max_length", value=_max_length)

            with st.form(key='my_form4'):
                text_input4 = st.text_area("Please enter the text to summarize:")
                submit_button4 = st.form_submit_button(label='Summarize the text')

            if submit_button4:
                bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
                bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
                input_tokenized = bart_tokenizer.encode(text_input4, return_tensors='pt').to(device)
                summary_ids = bart_model.generate(input_tokenized,
                                                  num_beams=_num_beams,
                                                  no_repeat_ngram_size=_no_repeat_ngram_size,
                                                  length_penalty=_length_penalty,
                                                  min_length=_min_length,
                                                  max_length=_max_length,
                                                  early_stopping=_early_stopping)

                m4_output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                          for g in summary_ids]
                st.write('Summary')
                st.success(m4_output[0])


if __name__ == '__main__':
    load = DataFrame_Loader()
    main()
