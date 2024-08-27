# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:19:34 2024

@author: aswan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 18:49:50 2024

@author: aswan
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import openai
import os

# Download nltk data
nltk.download('punkt')
nltk.download('stopwords')

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to split text into chunks
def split_text(text, max_length):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Function for LexRank Summarization using sumy
def lexrank_summary(text, num_sentences=10):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    lexrank_summarizer = LexRankSummarizer()
    summary = lexrank_summarizer(parser.document, sentences_count=num_sentences)
    return ' '.join([str(sentence) for sentence in summary])

# Function for TextRank Summarization
def textrank_summary(text, num_sentences=10):
    sentences = sent_tokenize(text)
    stop_words = stopwords.words('english')

    # Build the similarity matrix
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)

    # Build the graph and apply PageRank
    nx_graph = nx.from_numpy_array(cosine_matrix)
    scores = nx.pagerank(nx_graph)

    # Rank sentences based on scores
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Select the top n sentences for the summary
    summary = ' '.join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
    return summary

# Function for BART Summarization
def bart_summary(text):
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    chunks = split_text(text, 1024)  # Split text into chunks that BART can handle
    summaries = []

    for chunk in chunks:
        inputs = bart_tokenizer.encode(chunk, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = bart_model.generate(inputs)
        summaries.append(bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    
    return ' '.join(summaries)

# Function for T5 Summarization
def t5_summary(text):
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-large')
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-large')
    chunks = split_text(text, 1024)  # Split text into chunks that T5 can handle
    summaries = []

    for chunk in chunks:
        inputs = t5_tokenizer.encode("summarize: " + chunk, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = t5_model.generate(
            inputs, 
            max_length=150, 
            no_repeat_ngram_size=3)  # Prevent repetition
            
        
        summaries.append(t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    
    return ' '.join(summaries)
# Function for GPT-4 Summarization
def gpt4_summary(text):
    chunks = split_text(text, 4096)  # Split text into chunks that GPT-4 can handle
    summaries = []

    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a summarization assistant."},
                {"role": "user", "content": chunk},
            ],
        )
        summaries.append(response['choices'][0]['message']['content'])
    
    return ' '.join(summaries)

# Example usage:
if __name__ == "__main__":
    # Read the content from the file
    with open("text.txt", "r") as file:
        text = file.read()

    # LexRank Summary
    lexrank_result = lexrank_summary(text)
    print("LexRank Summary:")
    print(lexrank_result)
    
    # TextRank Summary
    textrank_result = textrank_summary(text)
    print("\nTextRank Summary:")
    print(textrank_result)
    
    # BART Summary
    bart_result = bart_summary(text)
    print("\nBART Summary:")
    print(bart_result)
    
    # T5 Summary
    t5_result = t5_summary(text)
    print("\nT5 Summary:")
    print(t5_result)
    
    # GPT-4 Summary
    gpt4_result = gpt4_summary(text)
    print("\nGPT-4 Summary:")
    print(gpt4_result)
