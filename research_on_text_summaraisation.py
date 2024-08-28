# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 12:40:09 2024

@author: aswan
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
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
import time
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import matplotlib.pyplot as plt

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

# Function to calculate ROUGE-1 scores
def calculate_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores['rouge1']

# Function to calculate METEOR score
def calculate_meteor(reference, summary):
    reference_tokens = nltk.word_tokenize(reference)
    summary_tokens = nltk.word_tokenize(summary)
    return meteor_score([reference_tokens], summary_tokens)

# Function to measure execution time
def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

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

    # Read the reference summary from the file
    with open("refe.txt", "r") as ref_file:
        reference_summary = ref_file.read()

    # Dictionaries to store results
    rouge_scores = {}
    meteor_scores = {}
    execution_times = {}
    summaries = {}

    # LexRank Summary
    lexrank_result, lexrank_time = measure_time(lexrank_summary, text)
    lexrank_rouge = calculate_rouge(reference_summary, lexrank_result)
    lexrank_meteor = calculate_meteor(reference_summary, lexrank_result)
    rouge_scores['LexRank'] = lexrank_rouge
    meteor_scores['LexRank'] = lexrank_meteor
    execution_times['LexRank'] = lexrank_time
    summaries['LexRank'] = lexrank_result
    
    # TextRank Summary
    textrank_result, textrank_time = measure_time(textrank_summary, text)
    textrank_rouge = calculate_rouge(reference_summary, textrank_result)
    textrank_meteor = calculate_meteor(reference_summary, textrank_result)
    rouge_scores['TextRank'] = textrank_rouge
    meteor_scores['TextRank'] = textrank_meteor
    execution_times['TextRank'] = textrank_time
    summaries['TextRank'] = textrank_result
    
    # BART Summary
    bart_result, bart_time = measure_time(bart_summary, text)
    bart_rouge = calculate_rouge(reference_summary, bart_result)
    bart_meteor = calculate_meteor(reference_summary, bart_result)
    rouge_scores['BART'] = bart_rouge
    meteor_scores['BART'] = bart_meteor
    execution_times['BART'] = bart_time
    summaries['BART'] = bart_result
    
    # T5 Summary
    t5_result, t5_time = measure_time(t5_summary, text)
    t5_rouge = calculate_rouge(reference_summary, t5_result)
    t5_meteor = calculate_meteor(reference_summary, t5_result)
    rouge_scores['T5'] = t5_rouge
    meteor_scores['T5'] = t5_meteor
    execution_times['T5'] = t5_time
    summaries['T5'] = t5_result
    
    # GPT-4 Summary
    gpt4_result, gpt4_time = measure_time(gpt4_summary, text)
    gpt4_rouge = calculate_rouge(reference_summary, gpt4_result)
    gpt4_meteor = calculate_meteor(reference_summary, gpt4_result)
    rouge_scores['GPT-4'] = gpt4_rouge
    meteor_scores['GPT-4'] = gpt4_meteor
    execution_times['GPT-4'] = gpt4_time
    summaries['GPT-4'] = gpt4_result
    
    # Print the summaries and response times
    print("\nSummaries and Response Times:")
    for model in summaries:
        print(f"\n{model} Summary:")
        print(summaries[model])
        print(f"Execution Time: {execution_times[model]:.2f} seconds")

    # Print the ROUGE-1 scores
    print("\nROUGE-1 Scores:")
    for model, scores in rouge_scores.items():
        print(f"{model}: Precision: {scores.precision:.4f}, Recall: {scores.recall:.4f}, F1: {scores.fmeasure:.4f}")
    
    # Print the METEOR scores
    print("\nMETEOR Scores:")
    for model, score in meteor_scores.items():
        print(f"{model}: METEOR: {score:.4f}")

    # Plot the ROUGE-1 scores
    models = list(rouge_scores.keys())
    precisions = [rouge_scores[model].precision for model in models]
    recalls = [rouge_scores[model].recall for model in models]
    fmeasures = [rouge_scores[model].fmeasure for model in models]

    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precisions, width, label='Precision')
    ax.bar(x, recalls, width, label='Recall')
    ax.bar(x + width, fmeasures, width, label='F1 Measure')

    ax.set_xlabel('Summarization Technique')
    ax.set_ylabel('ROUGE-1 Score')
    ax.set_title('ROUGE-1 Scores by Summarization Technique')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), title="Metrics")  # Adjust legend position

    plt.tight_layout()
    plt.show()

    # Plot the execution times
    times = [execution_times[model] for model in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(models, times, marker='o')

    for i, txt in enumerate(times):
        ax.annotate(f"{txt:.2f}s", (models[i], times[i]), textcoords="offset points", xytext=(0,-15), ha='center')  # Adjust annotation position

    ax.set_xlabel('Summarization Technique')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time by Summarization Technique')

    plt.tight_layout()
    plt.show()

    # Plot the METEOR scores
    meteor_values = [meteor_scores[model] for model in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.4
    ax.bar(models, meteor_values, color='orange', width = bar_width)

    ax.set_xlabel('Summarization Technique')
    ax.set_ylabel('METEOR Score')
    ax.set_title('METEOR Scores by Summarization Technique')

    plt.tight_layout()
    plt.show()

   
