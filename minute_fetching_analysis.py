import pandas as pd
import matplotlib.pyplot as plt
from bert_score import score

def calculate_bertscore(answer, reference_answer):
    """Calculate BERTScore between the answer and the reference answer."""
    P, R, F1 = score([answer], [reference_answer], lang="en", verbose=False)
    return F1.item()

def analyze_qa_data(file_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    
    # Calculate BERTScore for each row
    df['bertscore'] = df.apply(lambda row: calculate_bertscore(str(row['answer']), str(row['reference_answer'])), axis=1)
    
    # Calculate the average BERTScore
    average_bertscore = df['bertscore'].mean()
    
    # Print the average BERTScore
    print(f'Average BERTScore: {average_bertscore:.2%}')
    
    # Plotting the BERTScore bar graph
    plt.figure(figsize=(12, 7))
    plt.bar(df['id'], df['bertscore'], color='purple')
    plt.xlabel('Answer Instance (Query ID)', fontsize=14)
    plt.ylabel('BERTScore', fontsize=14)
    plt.title('BERTScore of Each Answer Compared to Reference Answer', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    # Plotting the response time comparison graph
    plt.figure(figsize=(12, 7))
    plt.plot(df['id'], df['response_time'], marker='o', label='First Response Time', color='blue')
    plt.plot(df['id'], df['second_response_time'], marker='o', label='Second Response Time', color='green')
    plt.xlabel('Query Instance (Query ID)', fontsize=14)
    plt.ylabel('Response Time (seconds)', fontsize=14)
    plt.title('Comparison of First and Second Response Times', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()


analyze_qa_data('qa_data.csv')
