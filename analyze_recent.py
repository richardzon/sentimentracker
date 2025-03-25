#!/usr/bin/env python3
"""
analyze_recent.py

A simple utility to analyze the sentiment of the most recent messages in the Discord database.
Uses OpenRouter's API (accessed via OpenAI client) to perform batch sentiment analysis.
"""

import os
import json
import sqlite3
import logging
import argparse
import asyncio
import time
import traceback
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from textblob import TextBlob  # Add TextBlob for fallback sentiment analysis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RecentSentimentAnalyzer')

# Load environment variables
load_dotenv()
DATABASE_PATH = os.getenv('DATABASE_PATH', '/home/richie/discord/discord_messages.db')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
SITE_URL = os.getenv('SITE_URL', 'https://bittensor.com')
SITE_NAME = os.getenv('SITE_NAME', 'Bittensor Sentiment Analysis')

if not OPENROUTER_API_KEY:
    logger.error("No OpenRouter API key found. Please set OPENROUTER_API_KEY in your .env file.")
    exit(1)

class RecentSentimentAnalyzer:
    """Analyzes sentiment of recent Discord messages in batch."""
    
    def __init__(self):
        """Initialize the analyzer with database connection."""
        self.conn = None
        self.cursor = None
        self.initialize_database()
        
    def initialize_database(self):
        """Connect to the existing SQLite database."""
        try:
            if not os.path.exists(DATABASE_PATH):
                logger.error(f"Database file not found: {DATABASE_PATH}")
                exit(1)
                
            self.conn = sqlite3.connect(DATABASE_PATH)
            self.conn.row_factory = sqlite3.Row  # Use dictionary-like rows
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {DATABASE_PATH}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
            
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
            
    async def analyze_recent_messages(self, limit=100, subnet_num=None):
        """
        Analyze sentiment of the most recent messages from the database.
        
        Args:
            limit: Number of recent messages to analyze (default 100)
            subnet_num: Optional subnet filter (if None, get messages from all subnets)
            
        Returns:
            A tuple containing (overall_sentiment, message_sentiments, mood_summary, key_topics)
        """
        logger.info(f"Analyzing sentiment of the most recent {limit} messages...")
        
        # Query to get the most recent messages
        if subnet_num is not None:
            self.cursor.execute("""
                SELECT id, content, author, timestamp, subnet_num 
                FROM messages 
                WHERE subnet_num = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (subnet_num, limit))
        else:
            self.cursor.execute("""
                SELECT id, content, author, timestamp, subnet_num 
                FROM messages 
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        recent_messages = self.cursor.fetchall()
        message_count = len(recent_messages)
        
        if message_count == 0:
            logger.info("No messages found.")
            return 0.0, [], "No messages to analyze", []
            
        logger.info(f"Found {message_count} messages for batch sentiment analysis")
        
        # Format messages for analysis
        batch_messages = []
        for msg in recent_messages:
            # Skip empty messages
            msg_id = msg['id']
            content = msg['content']
            author = msg['author']
            timestamp = msg['timestamp']
            subnet = msg['subnet_num']
            
            if not content or content.strip() == '':
                continue
                
            # Format for improved context
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                hour = dt.hour
                formatted_msg = f"{hour}:@{author}:{content}"
            except Exception as e:
                logger.warning(f"Error formatting timestamp {timestamp}: {e}")
                formatted_msg = f"@{author}:{content}"
            
            batch_messages.append({
                "id": msg_id,
                "content": formatted_msg,
                "raw_content": content,  # Keep raw content for TextBlob fallback
                "subnet": subnet
            })
        
        if not batch_messages:
            logger.info("No valid messages to analyze")
            return 0.0, [], "No valid messages to analyze", []
        
        # Try OpenRouter API with Qwen first
        results = await self._try_openrouter_analysis(batch_messages, model="qwen/qwq-32b")
        
        # If Qwen fails, try DeepSeek
        if not results:
            logger.info("Qwen model failed, trying DeepSeek model instead...")
            results = await self._try_openrouter_analysis(batch_messages, model="deepseek/deepseek-chat-v3-0324:free")
        
        # If both external models fail, fall back to TextBlob
        if not results:
            logger.info("External models failed, falling back to TextBlob for basic sentiment analysis...")
            results = self._analyze_with_textblob(batch_messages)
            
        return results
        
    async def _try_openrouter_analysis(self, batch_messages, model):
        """Try sentiment analysis using the specified OpenRouter model."""
        try:
            # Create a prompt with all messages
            messages_text = ""
            for i, msg in enumerate(batch_messages):
                messages_text += f"Message {i+1}. Subnet {msg['subnet']}: {msg['content']}\n\n"
            
            prompt = f"""
            You are analyzing sentiment in the most recent {len(batch_messages)} messages from Bittensor Discord channels.
            
            {messages_text}
            
            First, analyze EACH message individually:
            1. Determine the sentiment score from -1.0 (very negative) to 1.0 (very positive)
            2. Identify if it's a technical question or discussion (which should have more neutral sentiment)
            3. Note key topics mentioned
            
            Then, provide an OVERALL analysis:
            1. Overall sentiment score from -1.0 to 1.0 for all messages combined
            2. Brief summary of the general mood (1-2 sentences)
            3. Key trending topics in these messages
            4. Identify any significant positive or negative discussions
            
            Return a JSON object with these fields:
            - overall_sentiment: Score from -1.0 to 1.0
            - mood_summary: Brief text description
            - key_topics: Array of trending topics
            - message_sentiments: Array of objects with message_index (1-based), sentiment, is_technical, and key_topics
            
            Technical questions or discussions should have dampened sentiment (closer to zero).
            """
            
            # Set up OpenAI client with OpenRouter
            logger.info(f"Setting up OpenAI client for OpenRouter API")
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY
            )
            
            # Make the API call using the OpenAI client
            logger.info(f"Calling OpenRouter API ({model}) for batch sentiment analysis")
            start_time = time.time()
            
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": SITE_URL,
                    "X-Title": SITE_NAME
                },
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            # Get the response
            response_text = completion.choices[0].message.content.strip()
            logger.info(f"Batch sentiment analysis completed in {time.time() - start_time:.2f} seconds")
            
            # Print full response for debugging
            print(f"\n===== BEGIN RAW API RESPONSE FROM {model} =====")
            print(response_text)
            print("===== END RAW API RESPONSE =====\n")
            
            # Check if response is empty
            if not response_text:
                logger.warning("Empty response from OpenRouter API")
                return None
                
            try:
                # Clean up response if it contains markdown formatting
                if '```json' in response_text:
                    # Extract content between ```json and ```
                    start_idx = response_text.find('```json') + 7
                    end_idx = response_text.find('```', start_idx)
                    if end_idx > start_idx:
                        response_text = response_text[start_idx:end_idx].strip()
                elif '```' in response_text:
                    # Extract content between ``` and ```
                    start_idx = response_text.find('```') + 3
                    end_idx = response_text.find('```', start_idx)
                    if end_idx > start_idx:
                        response_text = response_text[start_idx:end_idx].strip()
                
                # Handle case where JSON might be embedded in explanatory text
                if '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        response_text = response_text[json_start:json_end]
                
                # Try multiple approaches to parse the JSON
                parsed = None
                try:
                    # Standard JSON parsing
                    parsed = json.loads(response_text)
                except json.JSONDecodeError as e:
                    # Try to fix common JSON issues
                    logger.warning(f"Initial JSON parsing failed: {e}")
                    
                    # Fix for trailing commas, which are common LLM errors
                    fixed_text = re.sub(r',\s*}', '}', response_text)
                    fixed_text = re.sub(r',\s*]', ']', fixed_text)
                    
                    # Fix for unquoted keys
                    fixed_text = re.sub(r'{\s*(\w+):', r'{"1": ', fixed_text)
                    fixed_text = re.sub(r',\s*(\w+):', r', "1": ', fixed_text)
                    
                    try:
                        parsed = json.loads(fixed_text)
                        logger.info("Successfully parsed JSON after fixing format issues")
                    except json.JSONDecodeError:
                        # Last resort: try to extract partial data
                        logger.warning("Trying to extract partial JSON data...")
                        
                        # Create default structure
                        parsed = {
                            "overall_sentiment": 0.0,
                            "mood_summary": "Unable to determine mood from API response",
                            "key_topics": [],
                            "message_sentiments": []
                        }
                        
                        # Try to extract overall sentiment
                        sentiment_match = re.search(r'"overall_sentiment":\s*([-+]?\d*\.\d+|\d+)', response_text)
                        if sentiment_match:
                            parsed["overall_sentiment"] = float(sentiment_match.group(1))
                            
                        # Try to extract mood summary
                        mood_match = re.search(r'"mood_summary":\s*"([^"]+)"', response_text)
                        if mood_match:
                            parsed["mood_summary"] = mood_match.group(1)
                            
                        # Try to extract some message sentiments
                        sentiment_items = re.findall(r'{"message_index":\s*(\d+),\s*"sentiment":\s*([-+]?\d*\.\d+|\d+)', response_text)
                        for idx, sentiment in sentiment_items:
                            parsed["message_sentiments"].append({
                                "message_index": int(idx),
                                "sentiment": float(sentiment),
                                "is_technical": False,  # Default assumption
                                "key_topics": []
                            })
                            
                        logger.info(f"Extracted partial data from malformed JSON: {len(parsed['message_sentiments'])} message sentiments")
                
                if not parsed:
                    logger.warning("Failed to parse JSON response")
                    return None
                
                # Extract overall sentiment and other data
                overall_sentiment = parsed.get('overall_sentiment', 0.0)
                mood_summary = parsed.get('mood_summary', '')
                key_topics = parsed.get('key_topics', [])
                message_sentiments = parsed.get('message_sentiments', [])
                
                # Log results
                logger.info(f"Overall sentiment of recent messages: {overall_sentiment:.2f}")
                logger.info(f"Mood summary: {mood_summary}")
                logger.info(f"Key topics: {', '.join(key_topics)}")
                logger.info(f"Analyzed {len(message_sentiments)} individual messages")
                
                return overall_sentiment, message_sentiments, mood_summary, key_topics
                
            except Exception as e:
                logger.error(f"Error processing API response: {e}")
                logger.error(traceback.format_exc())
                return None
                
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            logger.error(traceback.format_exc())
            return None
            
    def _analyze_with_textblob(self, batch_messages):
        """
        Analyze sentiment using TextBlob as a fallback method.
        
        This is a simpler analysis but provides a reliable baseline.
        """
        logger.info("Analyzing messages with TextBlob")
        
        # Analyze each message individually
        message_sentiments = []
        total_sentiment = 0.0
        technical_pattern = r'(?:how to|how do I|can someone help|error|bug|issue|problem|function|code|script|program|implementation|fix)'
        
        key_topics_set = set()
        
        for i, msg in enumerate(batch_messages):
            content = msg['raw_content']
            blob = TextBlob(content)
            sentiment = blob.sentiment.polarity
            
            # Check if message is technical
            is_technical = bool(re.search(technical_pattern, content.lower()))
            
            # Dampen sentiment for technical messages (closer to neutral)
            if is_technical:
                sentiment = sentiment * 0.5  # Reduce the magnitude by 50%
                
            # Extract possible topics (nouns)
            topics = []
            for word, tag in blob.tags:
                if tag.startswith('NN') and len(word) > 3:  # Nouns longer than 3 chars
                    topics.append(word.lower())
                    key_topics_set.add(word.lower())
            
            message_sentiments.append({
                "message_index": i + 1,
                "sentiment": sentiment,
                "is_technical": is_technical,
                "key_topics": topics[:3]  # Limit to top 3 topics per message
            })
            
            total_sentiment += sentiment
            
        # Calculate overall sentiment
        if message_sentiments:
            overall_sentiment = total_sentiment / len(message_sentiments)
        else:
            overall_sentiment = 0.0
            
        # Determine mood description
        if overall_sentiment > 0.2:
            mood = "The community shows a generally positive atmosphere."
        elif overall_sentiment < -0.2:
            mood = "The community displays signs of concern or negativity."
        else:
            mood = "The community shows a neutral or mixed sentiment overall."
            
        # Get top key topics
        key_topics = list(key_topics_set)[:10]  # Limit to top 10 topics
        
        logger.info(f"TextBlob analysis complete. Overall sentiment: {overall_sentiment:.2f}")
        return overall_sentiment, message_sentiments, mood, key_topics

    async def run(self, limit=100, subnet_num=None):
        """Run sentiment analysis and display formatted results."""
        try:
            subnet_str = f"subnet {subnet_num}" if subnet_num is not None else "all subnets"
            logger.info(f"Analyzing last {limit} messages from {subnet_str}...")
            
            # Run the analysis
            overall_sentiment, message_sentiments, mood_summary, key_topics = await self.analyze_recent_messages(
                limit=limit, 
                subnet_num=subnet_num
            )
            
            if not message_sentiments:
                print("No messages to analyze!")
                return
                
            # Determine overall mood description
            mood = "neutral"
            if overall_sentiment > 0.5:
                mood = "very positive"
            elif overall_sentiment > 0.1:
                mood = "positive"
            elif overall_sentiment < -0.5:
                mood = "very negative"
            elif overall_sentiment < -0.1:
                mood = "negative"
                
            # Count sentiment distribution
            positive_count = sum(1 for m in message_sentiments if m.get('sentiment', 0) > 0.1)
            negative_count = sum(1 for m in message_sentiments if m.get('sentiment', 0) < -0.1)
            neutral_count = len(message_sentiments) - positive_count - negative_count
            
            # Technical vs. non-technical
            technical_count = sum(1 for m in message_sentiments if m.get('is_technical', False))
            
            # Print results in a nicely formatted report
            print("\n" + "=" * 80)
            print(f"SENTIMENT ANALYSIS OF {len(message_sentiments)} RECENT MESSAGES FROM {subnet_str.upper()}")
            print("=" * 80)
            print(f"Overall sentiment: {overall_sentiment:.2f} ({mood})")
            print(f"Mood summary: {mood_summary}")
            print("-" * 80)
            print(f"Sentiment distribution: {positive_count} positive, {negative_count} negative, {neutral_count} neutral")
            print(f"Technical messages: {technical_count} ({technical_count/max(1,len(message_sentiments))*100:.1f}% of total)")
            print(f"Key topics: {', '.join(key_topics)}")
            print("-" * 80)
            
            # Show sample messages with highest/lowest sentiment
            if message_sentiments:
                # Sort by sentiment
                sorted_msgs = sorted(message_sentiments, key=lambda x: x.get('sentiment', 0))
                
                if sorted_msgs:
                    # Most negative messages (up to 3)
                    most_negative = sorted_msgs[:min(3, len(sorted_msgs))]
                    print("\nMost negative messages:")
                    for msg in most_negative:
                        idx = msg.get('message_index', 0)
                        sentiment = msg.get('sentiment', 0)
                        is_technical = "Technical" if msg.get('is_technical', False) else "Regular"
                        topics = ', '.join(msg.get('key_topics', []))
                        print(f"  Message {idx}: {sentiment:.2f} ({is_technical}) - Topics: {topics}")
                    
                    # Most positive messages (up to 3)
                    most_positive = sorted_msgs[-min(3, len(sorted_msgs)):]
                    most_positive.reverse()  # Show highest first
                    print("\nMost positive messages:")
                    for msg in most_positive:
                        idx = msg.get('message_index', 0)
                        sentiment = msg.get('sentiment', 0)
                        is_technical = "Technical" if msg.get('is_technical', False) else "Regular"
                        topics = ', '.join(msg.get('key_topics', []))
                        print(f"  Message {idx}: {sentiment:.2f} ({is_technical}) - Topics: {topics}")
            
            print("=" * 80)
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing recent messages: {e}")
            logger.error(traceback.format_exc())
            print(f"Error analyzing recent messages: {e}")
            return False

async def main():
    """Run the sentiment analysis."""
    parser = argparse.ArgumentParser(description='Analyze sentiment of recent Discord messages')
    parser.add_argument('--limit', type=int, default=100, help='Number of recent messages to analyze (default: 100)')
    parser.add_argument('--subnet', type=int, help='Filter messages to a specific subnet')
    args = parser.parse_args()
    
    analyzer = RecentSentimentAnalyzer()
    try:
        await analyzer.run(limit=args.limit, subnet_num=args.subnet)
    finally:
        analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())
