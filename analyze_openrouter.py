import pandas as pd
import os
import json
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenRouter API Key
OPENROUTER_API_KEY = "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"

def clean_messages(csv_path, max_messages=500):
    """Extract and clean the last messages from the CSV file."""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Sort by timestamp (most recent first)
    df['timestamp'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('timestamp', ascending=False)
    
    # Take the most recent messages
    recent_messages = df.head(max_messages)
    
    # Ensure content is string
    recent_messages['content'] = recent_messages['content'].astype(str)
    
    return recent_messages

def format_messages_for_analysis(messages_df):
    """Format messages for analysis in the format 'timestamp: user: message'."""
    formatted_messages = []
    
    for _, row in messages_df.iterrows():
        message = str(row['content'])
        author = str(row['author'])
        time_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Skip messages that are just 'nan' or empty
        if message.lower() == 'nan' or not message.strip():
            continue
            
        formatted_messages.append(f"{time_str}: {author}: {message}")
    
    return "\n".join(formatted_messages)

def analyze_with_openrouter(messages_text):
    """Send messages to OpenRouter for sentiment analysis using direct API call."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://discord-sentiment.ai",
        "X-Title": "Discord Message Sentiment Analyzer"
    }
    
    system_prompt = """You are an expert in sentiment analysis for technical communities. 
    Analyze these Discord messages from a technical community developing a subnet (a decentralized network for AI).
    
    Focus specifically on:
    1. Signs of excitement about upcoming news or releases
    2. Indications of features or updates that people are eagerly anticipating
    3. Future developments that community members are excited about
    4. Sentiment around potential upcoming changes
    
    Provide insights on:
    - What specific future developments or releases are being anticipated?
    - How excited are community members about these developments?
    - Are there any patterns in how these topics are discussed?
    - What is the overall sentiment regarding future developments?
    
    Format your response as a structured analysis with clear sections and include examples of messages that demonstrate anticipation or excitement about future developments."""

    data = {
        "model": "qwen/qwq-32b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here are the Discord messages to analyze:\n\n{messages_text[:8000]}"}  # Truncate to avoid token limits
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: No valid response from API. Full response: {json.dumps(result)}"
    except Exception as e:
        return f"Error during OpenRouter API call: {str(e)}\nFull response: {response.text if 'response' in locals() else 'No response'}"

def main():
    # Find the most recent CSV file in the scraped_data directory
    scraped_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scraped_data')
    csv_files = [f for f in os.listdir(scraped_data_dir) if f.startswith('discord_messages_') and f.endswith('.csv')]
    
    if not csv_files:
        print("No scraped data files found.")
        return
    
    # Sort by modification time (most recent first)
    csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(scraped_data_dir, x)), reverse=True)
    most_recent_file = os.path.join(scraped_data_dir, csv_files[0])
    
    print(f"Analyzing data from: {most_recent_file}")
    
    # Clean the messages
    messages_df = clean_messages(most_recent_file, max_messages=500)
    print(f"Extracted {len(messages_df)} messages for analysis")
    
    # Format messages for analysis
    print("Formatting messages for analysis...")
    formatted_messages = format_messages_for_analysis(messages_df)
    
    # Analyze with OpenRouter
    print("Sending to OpenRouter for analysis (Qwen 32B model)...")
    analysis = analyze_with_openrouter(formatted_messages)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(scraped_data_dir, f"openrouter_future_analysis_{timestamp}.txt")
    
    with open(output_path, 'w') as f:
        f.write(analysis)
    
    print(f"Analysis complete! Results saved to: {output_path}")
    print("\nAnalysis summary:")
    print("="*80)
    print(analysis)

if __name__ == "__main__":
    main()
