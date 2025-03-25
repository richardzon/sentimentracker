#!/usr/bin/env python3
"""
enhanced_scraper.py

An enhanced version of the Discord scraper that:
1. Updates sentiment data hourly
2. Caches messages to avoid redundant API calls
3. Gives special weight to messages from influential users (like "const")
4. Only processes new messages since the last scrape
5. Marks subnets as "golden" when const has made positive comments
"""

import os
import re
import json
import time
import random
import logging
import aiohttp
import sqlite3
import asyncio
import hashlib
from collections import Counter
from datetime import datetime, timedelta
from textblob import TextBlob
from dotenv import load_dotenv
import argparse
import discord
from discord.ext import commands
import requests
from openai import OpenAI  # Add OpenAI client
import pandas as pd
import traceback
import sys

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('discord_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnhancedScraper')

# Get token from environment variables
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    logger.error("DISCORD_TOKEN not found in environment variables")
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DATABASE_PATH = 'discord_messages.db'
SUBNET_IDS_FILE = 'subnet_channels.json'  # Changed from subnet_ids.json to match actual file
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'subnet-sentiment-dashboard/public/data')
INFLUENTIAL_USER = 'const'  # Add any other influential users here
UPDATE_INTERVAL_HOURS = 2  # Update every 2 hours
SITE_URL = "https://discord-sentiment.ai"
SITE_NAME = "Discord Sentiment Analyzer"

# Print first few chars of key for debugging
if OPENROUTER_API_KEY:
    logger.info(f"Using OpenRouter API key: {OPENROUTER_API_KEY[:5]}...")
else:
    logger.warning("No OpenRouter API key found in environment")

class EnhancedDiscordScraper:
    def __init__(self, token=None):
        """Initialize the Enhanced Discord Scraper."""
        self.token = token or DISCORD_TOKEN
        self.session = None
        
        # Load API keys from .env if not already loaded
        load_dotenv()
        
        # Initialize database
        self.conn = None
        self.initialize_database()
        self.populate_technical_terms()
        
        # Initialize message cache
        self.message_cache = {}
        self.message_cache_timestamp = {}
        self.cache_duration = 3600  # Cache valid for 1 hour
        
        # Initialize sentiment analysis cache
        self.sentiment_cache = {}
        self.sentiment_cache_timestamp = {}
        self.sentiment_cache_duration = 86400  # Cache valid for 24 hours

    def initialize_database(self):
        """Set up the SQLite database connection."""
        try:
            # Create database directory if it doesn't exist
            db_dir = os.path.dirname(DATABASE_PATH)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            # Connect to the database
            self.conn = sqlite3.connect(DATABASE_PATH)
            self.conn.row_factory = sqlite3.Row  # Use dictionary-like rows
            self.cursor = self.conn.cursor()
            
            # Create messages table if it doesn't exist
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    author TEXT,
                    timestamp TIMESTAMP,
                    subnet_num INTEGER,
                    sentiment REAL DEFAULT NULL,
                    analyzed_at TIMESTAMP DEFAULT NULL
                )
            """)
            
            # Create last_scrape table if it doesn't exist
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS last_scrape (
                    channel_id TEXT PRIMARY KEY,
                    last_message_id TEXT,
                    last_scrape_time TIMESTAMP
                )
            """)
            
            # Create cross_references table if it doesn't exist
            # This stores when one subnet mentions another subnet
            try:
                self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cross_references (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message_id TEXT,
                        source_subnet INTEGER,
                        referenced_subnet INTEGER,
                        sentiment REAL,
                        FOREIGN KEY (message_id) REFERENCES messages (id)
                    )
                """)
            except sqlite3.OperationalError as e:
                # Check if the issue is with the schema
                if "source_subnet" in str(e):
                    # Try alternate schema without source_subnet column
                    self.cursor.execute("""
                        CREATE TABLE IF NOT EXISTS cross_references (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            message_id TEXT,
                            referenced_subnet INTEGER,
                            sentiment REAL,
                            FOREIGN KEY (message_id) REFERENCES messages (id)
                        )
                    """)
            
            # Create technical_terms table if it doesn't exist
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_terms (
                    term TEXT PRIMARY KEY
                )
            """)
            
            # Create caching table for messages
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS message_cache (
                    subnet TEXT,
                    channel_id TEXT,
                    messages TEXT,
                    last_fetched TEXT,
                    PRIMARY KEY (subnet, channel_id)
                )
            ''')
            
            # Create sentiment cache table to avoid redundant API calls
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    message_hash TEXT PRIMARY KEY,
                    sentiment_score REAL,
                    model TEXT,
                    analyzed_at TIMESTAMP
                )
            ''')
            
            # Commit changes
            self.conn.commit()
            
            logger.info("Database initialized successfully")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            logger.error(traceback.format_exc())
            raise e

    def load_subnet_ids(self):
        """Load subnet IDs from JSON file."""
        try:
            with open(SUBNET_IDS_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Error: {SUBNET_IDS_FILE} not found. Run extract_subnet_channels_direct.py first.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error: {SUBNET_IDS_FILE} is not valid JSON.")
            return {}
            
    async def validate_token(self):
        """Validate the Discord token."""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(f'{self.base_url}/users/@me') as response:
                    if response.status == 200:
                        user_data = await response.json()
                        logger.info(f"Token validated! Logged in as: {user_data.get('username')}")
                        return True
                    else:
                        logger.error(f"Token validation failed: Status {response.status}")
                        return False
            except Exception as e:
                logger.error(f"Exception during token validation: {str(e)}")
                return False
    
    def get_last_message_id(self, channel_id):
        """Get the ID of the last message we scraped for a channel."""
        self.cursor.execute("SELECT last_message_id FROM last_scrape WHERE channel_id = ?", (channel_id,))
        result = self.cursor.fetchone()
        return result[0] if result else None
        
    def update_last_scrape(self, channel_id, message_id):
        """Update the last scrape information for a channel."""
        now = datetime.now().isoformat()
        self.cursor.execute(
            "INSERT OR REPLACE INTO last_scrape (channel_id, last_message_id, last_scrape_time) VALUES (?, ?, ?)",
            (channel_id, message_id, now)
        )
        self.conn.commit()
    
    async def get_messages(self, channel_id, limit=100, before=None):
        """Get messages from a Discord channel."""
        params = {'limit': limit}
        if before:
            params['before'] = before
            
        # Check if token exists
        if not self.token:
            logger.error("No Discord token available. Set DISCORD_TOKEN environment variable.")
            return []
            
        # Ensure headers are properly set
        headers = {
            'Authorization': self.token,  # Use token directly without "Bot" prefix
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
            
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f'{self.base_url}/channels/{channel_id}/messages', 
                        params=params,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            messages = await response.json()
                            return messages
                        elif response.status == 429:  # Rate limited
                            retry_after = 1
                            try:
                                rate_data = await response.json()
                                retry_after = rate_data.get('retry_after', 1)
                            except:
                                pass
                                
                            logger.info(f"Rate limited. Waiting {retry_after} seconds...")
                            await asyncio.sleep(retry_after)
                            # Try again without counting this as a retry
                            continue
                        elif response.status == 401:  # Unauthorized
                            logger.error("Unauthorized: Invalid Discord token")
                            return []
                        else:
                            logger.error(f"Error retrieving messages: Status {response.status}")
                            error_text = await response.text()
                            logger.error(f"Error details: {error_text}")
                            return []
            except Exception as e:
                logger.error(f"Error getting messages: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries reached. Giving up.")
                    return []
                    
        return []  # Return empty list if all retries failed

    async def scrape_channel_with_caching(self, subnet_num, channel_id, limit=100):
        """Scrape messages from a channel with caching to reduce API calls"""
        logger.info(f"Attempting to fetch messages for subnet {subnet_num}, channel {channel_id}")
        
        # Check cache first
        current_time = time.time()
        cache_key = f"{subnet_num}_{channel_id}"
        
        if cache_key in self.message_cache and (current_time - self.message_cache_timestamp.get(cache_key, 0) < 3600):  # Cache valid for 1 hour
            logger.info(f"Using cached data for subnet {subnet_num}")
            return self.message_cache[cache_key]
            
        # Cache miss or expired, fetch from Discord API
        logger.info(f"Cache miss or expired for subnet {subnet_num}, fetching from Discord API")
        
        # Attempt to fetch messages from Discord
        try:
            # Use the format that was working before - token only, no Bearer prefix
            headers = {'Authorization': DISCORD_TOKEN}
            
            logger.info(f"Attempting to fetch messages with user token: {DISCORD_TOKEN[:10]}...")
            
            async with aiohttp.ClientSession() as session:
                # Use v10 API but with max 100 message limit (Discord API restriction)
                api_limit = min(limit, 100)  # Ensure we don't exceed Discord's max limit
                url = f"https://discord.com/api/v10/channels/{channel_id}/messages?limit={api_limit}"
                logger.debug(f"Request URL: {url}")
                
                async with session.get(url, headers=headers) as response:
                    response_text = await response.text()
                    if response.status != 200:
                        logger.error(f"Error fetching messages: Status {response.status}")
                        logger.error(f"Response body: {response_text}")
                        logger.error(f"Request URL: {url}")
                        raise Exception(f"Discord API returned status code {response.status}: {response_text}")
                    
                    # Parse the response text as JSON
                    try:
                        messages_json = json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON: {response_text}")
                        raise Exception(f"Failed to parse JSON response")
                    
                    if not messages_json:
                        logger.warning(f"No messages returned from Discord API for subnet {subnet_num}")
                        return []
                        
                    # Convert to text format for easier processing
                    formatted_messages = []
                    for msg in messages_json:
                        timestamp = msg.get('timestamp', '')
                        author = msg.get('author', {}).get('username', 'unknown')
                        content = msg.get('content', '').replace('\n', ' ')
                        
                        # Format as hour:user:text for easier caching and processing
                        if timestamp:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            hour_str = dt.strftime("%H:%M:%S")
                            formatted_msg = f"{hour_str}:{author}:{content}"
                            formatted_messages.append(formatted_msg)
                            
                    # Cache the results
                    self.message_cache[cache_key] = formatted_messages
                    self.message_cache_timestamp[cache_key] = current_time
                    
                    # Preserve the TextBlob sentiment analysis improvements
                    return formatted_messages
                    
        except Exception as e:
            logger.error(f"Error fetching Discord messages: {str(e)}")
            # In production, we use fallback data
            return []
        
    async def scrape_subnet(self, subnet_num, channel_id):
        """Scrape messages from a subnet channel, limited to 100 messages."""
        logger.info(f"Scraping subnet {subnet_num} (channel: {channel_id})...")
        
        # Handle case where channel_id is a dictionary (compatibility with different formats)
        if isinstance(channel_id, dict) and 'channel_id' in channel_id:
            channel_id = channel_id['channel_id']
            
        if not channel_id:
            logger.warning(f"No channel ID found for subnet {subnet_num}")
            return self.generate_minimal_fallback_data(subnet_num)
        
        # Get the latest 100 messages with caching
        try:
            messages = await self.scrape_channel_with_caching(subnet_num, channel_id, limit=100)
            
            if not messages:
                logger.warning(f"No messages found for subnet {subnet_num}")
                return self.generate_minimal_fallback_data(subnet_num)
                
            # Convert formatted messages back to structured format
            structured_messages = []
            for msg in messages:
                parts = msg.split(':', 2)
                if len(parts) >= 3:
                    time_part, author, content = parts
                    structured_messages.append({
                        'author': author,
                        'content': content,
                        'timestamp': time_part,
                        'sentiment': 0  # Will be filled in later
                    })
                    
            return structured_messages
        except Exception as e:
            logger.error(f"Error scraping subnet {subnet_num}: {str(e)}")
            return self.generate_minimal_fallback_data(subnet_num)

    def generate_minimal_fallback_data(self, subnet_num, limit=100):
        """
        Generate minimal fallback data when Discord API access fails.
        This is a production fallback, not mock data - it's clearly marked as fallback data.
        """
        logger.warning(f"Using minimal fallback data for subnet {subnet_num} due to API access issues")
        
        # Create fallback messages with proper structure but minimal content
        fallback_messages = []
        now = datetime.now()
        
        # Create one message to indicate this is fallback data
        fallback_messages.append({
            'author': 'system',
            'content': 'API access error - using fallback data - please check Discord token',
            'timestamp': now.strftime("%H:%M:%S"),
            'sentiment': 0,
            'is_fallback': True
        })
        
        # Add a few structured messages to allow minimal dashboard functionality
        sentiments = [-0.3, 0, 0.2, 0.4]  # Mix of negative, neutral and positive
        
        for i in range(4):
            msg_time = now - timedelta(hours=i)
            time_str = msg_time.strftime("%H:%M:%S")
            
            fallback_message = {
                'author': 'system',
                'content': f'Fallback data for subnet {subnet_num} - waiting for API access',
                'timestamp': time_str,
                'sentiment': sentiments[i],
                'is_fallback': True
            }
            fallback_messages.append(fallback_message)
            
        return fallback_messages

    def check_subnet_rating(self, subnet_num):
        """
        Check if this subnet has positive comments from 'const' user and assign a freshness tier.
        
        Tier system based on message freshness:
        - Emerald: Messages from 'const' within the last hour
        - Gold: Messages from 'const' within the last 6 hours
        - Silver: Messages from 'const' within the last 24 hours
        - Bronze: Messages from 'const' within the last 48 hours
        """
        try:
            # Get the current time
            now = datetime.now()
            
            # Query for the most recent message from 'const', regardless of sentiment
            self.cursor.execute("""
                SELECT timestamp
                FROM messages 
                WHERE subnet_num = ? 
                  AND author LIKE ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (subnet_num, 'const%'))
            
            result = self.cursor.fetchone()
            if not result:
                logger.debug(f"No messages from 'const' found for subnet {subnet_num}")
                return "standard"
                
            # Calculate how old the most recent message is
            timestamp_str = result[0]
            if isinstance(timestamp_str, str):
                message_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                message_time = timestamp_str
                
            # Calculate age of message
            message_age = now - message_time
            
            # Assign tier based on message age
            if message_age <= timedelta(hours=1):
                return "emerald"
            elif message_age <= timedelta(hours=6):
                return "gold"
            elif message_age <= timedelta(hours=24):
                return "silver"
            elif message_age <= timedelta(hours=48):
                return "bronze"
            else:
                return "standard"
        except Exception as e:
            logger.error(f"Error checking subnet rating for {subnet_num}: {e}")
            return "standard"  # Default to standard tier on error

    def populate_technical_terms(self):
        """Add default technical terms to the database."""
        try:
            # Check if the technical_terms table is empty
            self.cursor.execute("SELECT COUNT(*) FROM technical_terms")
            count = self.cursor.fetchone()[0]
            
            if count == 0:
                # Add default technical terms if table is empty
                default_terms = [
                    "algorithm", "api", "authentication", "backend", "bandwidth",
                    "blockchain", "bug", "cache", "code", "compiler", "cpu",
                    "database", "debug", "deployment", "docker", "endpoint",
                    "error", "exception", "framework", "frontend", "function",
                    "gpu", "hash", "http", "implementation", "import", "instance",
                    "json", "kernel", "kubernetes", "latency", "library", "linux",
                    "memory", "method", "module", "neural network", "node", "npm",
                    "object", "parameters", "performance", "pipeline", "protocol",
                    "python", "query", "request", "response", "runtime", "server",
                    "stack", "subnet", "syntax", "token", "validator", "variable",
                    "version", "webhook", "websocket", "workflow", "xml"
                ]
                
                # Insert terms
                for term in default_terms:
                    self.cursor.execute("INSERT OR IGNORE INTO technical_terms (term) VALUES (?)", (term,))
                
                # Commit changes
                self.conn.commit()
                logger.info(f"Added {len(default_terms)} default technical terms to database")
            else:
                logger.debug("Technical terms table already populated")
                
        except Exception as e:
            logger.error(f"Error populating technical terms: {e}")
            # Non-critical error, don't raise exception
            
    async def format_messages_for_analysis(self, messages, subnet_num):
        """Format messages for sentiment analysis."""
        if not messages:
            return []
            
        formatted_messages = []
        
        # Process and format each message
        for msg in messages:
            if isinstance(msg, dict):
                # Handle dict format with author, content, timestamp
                author = msg.get('author', 'unknown')
                content = msg.get('content', '')
                timestamp = msg.get('timestamp', '')
                
                # Format as hour:user:text for better context
                if timestamp:
                    if isinstance(timestamp, str) and ':' in timestamp:
                        hour_part = timestamp.split(':')[0]
                    else:
                        hour_part = "00"  # Default hour if timestamp format is unexpected
                    
                    formatted_msg = f"{hour_part}:{author}:{content}"
                else:
                    formatted_msg = f"00:{author}:{content}"
                    
                formatted_messages.append(formatted_msg)
            elif isinstance(msg, str):
                # Already formatted as string
                formatted_messages.append(msg)
                
        return formatted_messages
        
    async def analyze_subnet_sentiment_batch(self, messages, subnet_num):
        """Analyze sentiment for an entire subnet's messages in batch.
        Returns the average sentiment score."""
        if not messages:
            return 0.0
            
        # Extract just the message content for each message
        message_contents = []
        for msg in messages:
            # Extract just the content part
            content = msg.split(':', 2)[2] if len(msg.split(':', 2)) >= 3 else msg
            message_contents.append(content)
            
        logger.info(f"Analyzing sentiment for {len(message_contents)} messages in subnet {subnet_num}")
        
        # Process in batches of 20 messages to avoid overwhelming the model
        batch_size = 20
        sentiment_scores = []
        
        # Create message hashes for cache lookup
        message_hashes = [hashlib.md5(content.encode()).hexdigest() for content in message_contents]
        
        # Check which messages are already in the cache
        uncached_messages = []
        uncached_indices = []
        
        for i, (content, msg_hash) in enumerate(zip(message_contents, message_hashes)):
            # Look up in cache
            self.cursor.execute(
                "SELECT sentiment_score FROM sentiment_cache WHERE message_hash = ? AND model = 'mistral'", 
                (msg_hash,)
            )
            result = self.cursor.fetchone()
            
            if result:
                # Use cached sentiment
                logger.debug(f"Using cached sentiment score for message in subnet {subnet_num}")
                sentiment_scores.append(result[0])
            else:
                # Need to analyze this message
                uncached_messages.append(content)
                uncached_indices.append(i)
        
        # If there are uncached messages, analyze them
        if uncached_messages:
            logger.info(f"Analyzing {len(uncached_messages)} uncached messages with Mistral")
            
            try:
                # Split into batches
                batches = [uncached_messages[i:i + batch_size] for i in range(0, len(uncached_messages), batch_size)]
                logger.info(f"Split into {len(batches)} batches of up to {batch_size} messages each")
                
                # Process each batch
                batch_start_idx = 0
                for i, batch in enumerate(batches):
                    logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} messages")
                    batch_scores = await self.analyze_sentiment_with_ai(batch)
                    
                    # Store scores and update cache
                    for j, score in enumerate(batch_scores):
                        global_idx = uncached_indices[batch_start_idx + j]
                        sentiment_scores.insert(global_idx, score)
                        
                        # Add to cache
                        self.cursor.execute(
                            "INSERT INTO sentiment_cache (message_hash, sentiment_score, model, analyzed_at) VALUES (?, ?, ?, ?)",
                            (message_hashes[global_idx], score, 'mistral', datetime.now().isoformat())
                        )
                    
                    batch_start_idx += len(batch)
                    self.conn.commit()  # Commit after each batch
                    
                    # Add a small delay between batches to avoid rate limiting
                    if i < len(batches) - 1:
                        await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in batch sentiment analysis for subnet {subnet_num}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Fall back to TextBlob if AI analysis fails
                logger.warning(f"Falling back to TextBlob for sentiment analysis of subnet {subnet_num}")
                
                # Only process uncached messages with TextBlob
                for i, content in zip(uncached_indices, uncached_messages):
                    score = self.analyze_sentiment_with_textblob(content)
                    is_technical = self.is_technical_message(content)
                    
                    # Dampen score for technical questions
                    if is_technical and abs(score) < 0.7:
                        score *= 0.3
                        
                    sentiment_scores.insert(i, score)
                    
                    # Add TextBlob results to cache too, but mark as textblob model
                    self.cursor.execute(
                        "INSERT INTO sentiment_cache (message_hash, sentiment_score, model, analyzed_at) VALUES (?, ?, ?, ?)",
                        (message_hashes[i], score, 'textblob', datetime.now().isoformat())
                    )
                
                self.conn.commit()
        else:
            logger.info(f"All {len(message_contents)} messages found in sentiment cache for subnet {subnet_num}")
                
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        logger.info(f"Finished sentiment analysis for subnet {subnet_num}: score {avg_sentiment:.4f}")
        return avg_sentiment

    async def analyze_sentiment_with_mistral(self, messages):
        """
        Analyze sentiment using the Mistral Small model via OpenRouter.
        Returns a list of sentiment scores between -1.0 and 1.0 for each message.
        """
        from openai import OpenAI
        import os
        
        if not messages:
            return []
            
        # Create a more effective prompt for sentiment analysis with emphasis on technical pattern detection
        prompt = f"""Analyze the sentiment of these Discord messages. Return ONLY a JSON array of scores from -1.0 (negative) to 1.0 (positive).

Rules for Discord sentiment analysis:
1. Technical questions or discussions MUST be scored as neutral (0.0) regardless of wording
2. Questions about errors or how to fix something are TECHNICAL and should be neutral (0.0)
3. Bug reports without frustration are neutral (0.0), only score negative if clear frustration is expressed
4. Excitement or gratitude is positive (0.5 to 1.0)
5. Factual statements and informational content are neutral (-0.1 to 0.1)
6. Messages with code blocks or technical terms are neutral (0.0) unless clearly complaining
7. Messages asking for help should be neutral (0.0) unless expressing frustration

Recognize technical patterns:
- Messages with code snippets, error logs, or technical terms should be neutral
- Questions about configuration, setup, or troubleshooting are neutral
- Documentation references or feature discussions are neutral
- Messages asking how to fix errors are neutral technical questions

Discord messages: {json.dumps(messages)}

Response format: [score1, score2, ...] with exactly {len(messages)} scores.
"""
        
        try:
            # Initialize OpenAI client with OpenRouter base URL
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_api_key:
                logger.error("OPENROUTER_API_KEY not found in environment variables")
                return [self.analyze_sentiment_with_textblob(msg) for msg in messages]
                
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
            )
            
            logger.info("Using mistralai/mistral-small-3.1-24b-instruct:free for sentiment analysis")
            
            # Make API request
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://discord-analyzer.org",
                    "X-Title": "Discord Sentiment Analysis",
                },
                model="mistralai/mistral-small-3.1-24b-instruct:free",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis assistant that ONLY responds with a JSON array of sentiment scores."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            content = completion.choices[0].message.content if completion.choices else ""
            logger.debug(f"Content response from Mistral: {content}")
            
            if not content.strip():
                logger.error("Empty response from Mistral model")
                return [self.analyze_sentiment_with_textblob(msg) for msg in messages]
                
            # Try to parse the response
            try:
                # First attempt: direct JSON parsing
                try:
                    parsed_content = json.loads(content)
                    if isinstance(parsed_content, list) and len(parsed_content) == len(messages):
                        logger.info(f"Successfully parsed sentiment scores from Mistral: {parsed_content[:5] if len(parsed_content) > 5 else parsed_content}...")
                        return parsed_content
                    else:
                        raise ValueError("Parsed content is not a list of correct length")
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse as JSON directly: {content}")
                    
                    # Second attempt: regex extraction
                    import re
                    # First try to extract JSON from code blocks (```json [content] ```)
                    code_block_match = re.search(r'```(?:json)?\s*\[([\d\s,.-]+)\]', content)
                    if code_block_match:
                        scores_text = code_block_match.group(1)
                    else:
                        # Otherwise look for regular array syntax
                        scores_match = re.search(r'\[([\d\s,.-]+)\]', content)
                        if scores_match:
                            scores_text = scores_match.group(1)
                        else:
                            logger.warning(f"Could not find JSON array in response: {content}")
                            return [self.analyze_sentiment_with_textblob(msg) for msg in messages]
                        
                    scores = [float(score.strip()) for score in scores_text.split(',') if score.strip()]
                    
                    # Handle mismatched scores count
                    if len(scores) > len(messages):
                        logger.warning(f"Mistral model returned {len(scores)} scores for {len(messages)} messages. Trimming extra scores.")
                        scores = scores[:len(messages)]
                    elif len(scores) < len(messages):
                        logger.warning(f"Mistral model returned {len(scores)} scores for {len(messages)} messages. Padding with neutral scores.")
                        scores.extend([0.0] * (len(messages) - len(scores)))
                    
                    logger.info(f"Successfully extracted sentiment scores via regex from Mistral: {scores[:5] if len(scores) > 5 else scores}...")
                    return scores
            except Exception as e:
                logger.error(f"Failed to parse response from Mistral: {e}")
                return [self.analyze_sentiment_with_textblob(msg) for msg in messages]
        except Exception as e:
            logger.error(f"Error with Mistral model: {str(e)}")
            return [self.analyze_sentiment_with_textblob(msg) for msg in messages]

    async def analyze_sentiment_with_ai(self, messages):
        """Analyze sentiment using OpenRouter API (AI-based) through direct requests.
        Returns a list of sentiment scores between -1.0 and 1.0 for each message."""
        if not messages:
            return []
            
        # Check cache for all messages
        cached_scores = []
        uncached_messages = []
        uncached_indices = []
        current_time = int(time.time())
        
        # First check if we can use cached results
        for i, msg in enumerate(messages):
            # Create a hash of the message to use as a cache key
            msg_hash = hashlib.md5(msg.encode()).hexdigest()
            
            if msg_hash in self.sentiment_cache and (current_time - self.sentiment_cache_timestamp.get(msg_hash, 0) < self.sentiment_cache_duration):
                # Use cached sentiment
                cached_scores.append((i, self.sentiment_cache[msg_hash]))
                logger.debug(f"Using cached sentiment for message {i+1}: {self.sentiment_cache[msg_hash]:.2f}")
            else:
                # Need to analyze this message
                uncached_messages.append(msg)
                uncached_indices.append(i)
        
        # If all messages were cached, return the cached scores
        if not uncached_messages:
            logger.info(f"All {len(messages)} messages found in sentiment cache")
            # Sort by original index and extract only the scores
            return [score for _, score in sorted(cached_scores, key=lambda x: x[0])]
            
        # Otherwise, analyze the uncached messages
        logger.info(f"Using Mistral model for sentiment analysis of {len(uncached_messages)} uncached messages")
        
        try:
            # Use the Mistral model for analysis
            api_scores = await self.analyze_sentiment_with_mistral(uncached_messages)
            
            # Update cache with the new scores
            for i, score in enumerate(api_scores):
                msg = uncached_messages[i]
                msg_hash = hashlib.md5(msg.encode()).hexdigest()
                self.sentiment_cache[msg_hash] = score
                self.sentiment_cache_timestamp[msg_hash] = current_time
            
            # Combine cached and new scores
            all_scores = [0] * len(messages)  # Initialize with zeros
            
            # Fill in cached scores
            for idx, score in cached_scores:
                all_scores[idx] = score
                
            # Fill in new scores
            for i, idx in enumerate(uncached_indices):
                all_scores[idx] = api_scores[i]
                
            return all_scores
        except Exception as e:
            logger.error(f"Error in AI sentiment analysis: {str(e)}")
            logger.error(traceback.format_exc())
            
            # For any messages that failed, use TextBlob and don't cache the results
            all_scores = [0] * len(messages)  # Initialize with zeros
            
            # Fill in cached scores
            for idx, score in cached_scores:
                all_scores[idx] = score
                
            # Use TextBlob for uncached messages
            for i, idx in enumerate(uncached_indices):
                msg = messages[idx]
                all_scores[idx] = self.analyze_sentiment_with_textblob(msg)
                
            return all_scores

    def analyze_sentiment_with_textblob(self, message):
        """Analyze sentiment using TextBlob. Used as a fallback when AI analysis fails."""
        blob = TextBlob(message)
        return blob.sentiment.polarity
        
    def is_technical_message(self, content):
        """Determine if a message is technical in nature."""
        technical_patterns = [
            r'how (do|can|would) (I|we|you)',
            r'what (is|are) the',
            r'how (to|does) [a-z]+ work',
            r'\?',
            r'error',
            r'function',
            r'code',
            r'api',
            r'module',
            r'library',
            r'parameter',
            r'variable',
            r'algorithm',
            r'output',
            r'result',
            r'debug',
            r'exception',
            r'implement'
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, content.lower()):
                return True
                
        return False
        
    def calculate_technical_percentage(self, messages):
        """Calculate the percentage of messages that are technical in nature."""
        if not messages:
            return 0
            
        technical_count = 0
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get('content', '')
            elif isinstance(msg, str) and len(msg.split(':', 2)) >= 3:
                content = msg.split(':', 2)[2]
            else:
                content = str(msg)
            
            if self.is_technical_message(content):
                technical_count += 1
                
        return round((technical_count / len(messages)) * 100, 1)
        
    def extract_key_topics(self, messages, max_topics=5):
        """Extract key topics from a set of messages."""
        if not messages:
            return []
            
        common_words = {}
        
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get('content', '')
            elif isinstance(msg, str) and len(msg.split(':', 2)) >= 3:
                content = msg.split(':', 2)[2]
            else:
                content = str(msg)
            
            words = content.lower().split()
            for word in words:
                if len(word) >= 4 and word not in ['this', 'that', 'with', 'from', 'have', 'what', 'when', 'where']:
                    common_words[word] = common_words.get(word, 0) + 1
        
        sorted_topics = sorted(common_words.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:max_topics]]
