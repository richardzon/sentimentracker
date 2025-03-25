import discord
import aiohttp
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import asyncio
import sys
import json
import re
import traceback

# Load environment variables
load_dotenv()

class BittensorDiscordScraper:
    def __init__(self, token):
        self.token = token
        self.base_url = 'https://discord.com/api/v9'
        self.headers = {
            'Authorization': token,
            'Content-Type': 'application/json'
        }
        self.messages_data = []
        self.target_roles = []
        self.guild_roles = {}
        self.member_roles = {}
        
    async def validate_token(self):
        """Validate the token by making a request to get the user's information."""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(f'{self.base_url}/users/@me') as response:
                    if response.status == 200:
                        user_data = await response.json()
                        print(f"Token validated successfully! Logged in as: {user_data.get('username')}#{user_data.get('discriminator')}")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"Token validation failed: Status {response.status}")
                        print(f"Error details: {error_text}")
                        return False
            except Exception as e:
                print(f"Exception during token validation: {str(e)}")
                return False
        
    async def get_guilds(self):
        """Get list of guilds (servers) the user is in."""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(f'{self.base_url}/users/@me/guilds') as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    raise Exception("Invalid token. Make sure you're using your user token.")
                else:
                    raise Exception(f"Failed to get guilds: {response.status}")

    async def get_guild_roles(self, guild_id):
        """Get all roles in a guild."""
        if guild_id not in self.guild_roles:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(f'{self.base_url}/guilds/{guild_id}/roles') as response:
                    if response.status == 200:
                        roles = await response.json()
                        # Sort roles by position (higher position = more important)
                        roles.sort(key=lambda x: x['position'], reverse=True)
                        self.guild_roles[guild_id] = roles
                    else:
                        self.guild_roles[guild_id] = []
        return self.guild_roles[guild_id]

    def display_and_select_roles(self, roles):
        """Display available roles and let user select which ones to scrape."""
        print("\nAvailable roles:")
        for i, role in enumerate(roles, 1):
            print(f"{i}. {role['name']}")
        
        print("\nEnter the numbers of the roles you want to scrape (comma-separated)")
        print("Example: 1,3,5 or just press Enter to scrape all roles")
        
        while True:
            try:
                selection = input("> ").strip()
                if not selection:  # Empty input means all roles
                    return [role['id'] for role in roles]
                    
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_roles = []
                
                for idx in indices:
                    if 0 <= idx < len(roles):
                        selected_roles.append(roles[idx]['id'])
                    else:
                        print(f"Invalid selection: {idx + 1}")
                        continue
                
                if selected_roles:
                    print("\nSelected roles:")
                    for role_id in selected_roles:
                        role_name = next(r['name'] for r in roles if r['id'] == role_id)
                        print(f"- {role_name}")
                    return selected_roles
                    
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
            except Exception as e:
                print(f"Error: {str(e)}")
            
            print("\nPlease try again:")
            
    async def get_member_roles(self, guild_id, user_id):
        """Get roles for a specific member."""
        cache_key = f"{guild_id}_{user_id}"
        if cache_key not in self.member_roles:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(f'{self.base_url}/guilds/{guild_id}/members/{user_id}') as response:
                    if response.status == 200:
                        member_data = await response.json()
                        self.member_roles[cache_key] = member_data.get('roles', [])
                    else:
                        self.member_roles[cache_key] = []
        return self.member_roles[cache_key]

    async def should_include_message(self, message, guild_id):
        """Check if a message should be included based on filters."""
        # For the specific channel we want all messages, so return True
        return True

    async def get_channels(self, guild_id):
        """Get list of channels in a guild."""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(f'{self.base_url}/guilds/{guild_id}/channels') as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Failed to get channels for guild {guild_id}: {response.status}")
                    return []

    async def check_channel_access(self, channel_id):
        """Check if the user has access to a specific channel."""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(f'{self.base_url}/channels/{channel_id}') as response:
                    if response.status == 200:
                        channel_data = await response.json()
                        print(f"Channel access verified: {channel_data.get('name')} (ID: {channel_id})")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"Cannot access channel {channel_id}: Status {response.status}")
                        print(f"Error details: {error_text}")
                        return False
            except Exception as e:
                print(f"Exception when checking channel access: {str(e)}")
                return False

    async def get_messages(self, channel_id, limit=100, before=None):
        """Get messages from a channel."""
        params = {'limit': limit}
        if before:
            params['before'] = before
            
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(f'{self.base_url}/channels/{channel_id}/messages', params=params) as response:
                    if response.status == 200:
                        messages = await response.json()
                        print(f"Successfully retrieved {len(messages)} messages from channel {channel_id}")
                        return messages
                    else:
                        error_text = await response.text()
                        print(f"Error retrieving messages from channel {channel_id}: Status {response.status}")
                        print(f"Error details: {error_text}")
                        if response.status == 403:
                            print("Access Forbidden: Your token may not have permission to view this channel")
                        elif response.status == 404:
                            print("Channel Not Found: The channel ID may be incorrect or the channel doesn't exist")
                        elif response.status == 429:
                            print("Rate Limited: Discord API rate limit has been reached")
                        elif response.status == 401:
                            print("Unauthorized: Your Discord token is invalid or expired. Please update it in your .env file.")
                            print("\nTo get a valid Discord token:")
                            print("1. Open Discord in your web browser")
                            print("2. Press F12 to open Developer Tools")
                            print("3. Go to the Network tab")
                            print("4. Send a message or interact with Discord")
                            print("5. Look for a request to 'discord.com'")
                            print("6. In the request headers, find 'Authorization'")
                            print("7. Copy the token value and add it to your .env file as DISCORD_TOKEN=your_token")
                        return []
            except Exception as e:
                print(f"Exception when retrieving messages: {str(e)}")
                return []

    def analyze_sentiment(self, messages_batch):
        """
        Analyze sentiment of messages using TextBlob, a machine learning based
        sentiment analysis library. This provides more accurate results than 
        a simple keyword-based approach, especially for technical discussions.
        Returns a list of sentiment scores from VERY_NEGATIVE (-2) to VERY_POSITIVE (2).
        """
        try:
            print(f"Analyzing sentiment for {len(messages_batch)} messages...")
            
            # Check if TextBlob is installed
            try:
                from textblob import TextBlob
            except ImportError:
                print("TextBlob not installed. Installing now...")
                import subprocess
                subprocess.check_call(['pip', 'install', 'textblob'])
                from textblob import TextBlob
                
                # Download necessary NLTK data for TextBlob
                import nltk
                nltk.download('punkt')
            
            # Sentiment categories and scores
            sentiment_map = {
                -2: "VERY_NEGATIVE",
                -1: "NEGATIVE",
                0: "NEUTRAL",
                1: "POSITIVE",
                2: "VERY_POSITIVE"
            }
            
            # Track user sentiment history for context
            if not hasattr(self, 'user_sentiment_history'):
                self.user_sentiment_history = {}
            
            # Domain-specific context words (not for scoring, but for context)
            subnet_context_words = [
                "subnet", "validator", "miner", "mining", "stake", "staking", 
                "reward", "incentive", "token", "tao", "emissions", "weights", 
                "scores", "registration", "consensus", "network", "community"
            ]
            
            # Define regex patterns for questions 
            question_patterns = [
                r'\?$',                  # Ends with question mark
                r'^(who|what|when|where|why|how|is|are|can|could|would|will|should|did|do|does)'  # Starts with question word
            ]
            
            # Define regex patterns for technical observations
            technical_patterns = [
                r'(uid|validator|miner|subnet)\s+\d+',  # References to UIDs, validators, miners with numbers
                r'(0x[a-fA-F0-9]{40})',  # Ethereum addresses
                r'(5[a-zA-Z0-9]{47})',   # Substrate addresses (like 5HbScNssaEfioJHXjcXdpyqo...)
            ]
            
            results = []
            
            # Process each message for sentiment
            for message in messages_batch:
                content = message['content'].lower()
                author_id = message['author']['id']
                
                if not content.strip():  # Skip empty messages
                    results.append({
                        'message_id': message['id'],
                        'sentiment_score': 0,
                        'sentiment_label': "NEUTRAL",
                        'raw_score': 0
                    })
                    continue
                
                # Check if message is a question or technical observation
                is_question = any(re.search(pattern, content) for pattern in question_patterns)
                is_technical = any(re.search(pattern, content) for pattern in technical_patterns)
                
                # Use TextBlob for sentiment analysis
                blob = TextBlob(content)
                
                # TextBlob provides polarity between -1 and 1
                # We'll scale it to our range (-2 to 2)
                raw_score = blob.sentiment.polarity * 2
                
                # Apply special rules for technical discussions
                # Technical questions and observations without clear sentiment should be neutral
                if is_question:
                    # Questions about technical topics should almost always be neutral
                    if any(word in content for word in subnet_context_words):
                        raw_score = 0  # Force technical questions to be neutral
                    elif abs(raw_score) < 0.8:  # Higher threshold for questions
                        raw_score = 0
                elif is_technical and abs(raw_score) < 0.5:
                    raw_score = 0
                
                # Track user sentiment history for context
                if author_id not in self.user_sentiment_history:
                    self.user_sentiment_history[author_id] = []
                
                # Keep only the last 5 messages for history
                if len(self.user_sentiment_history[author_id]) >= 5:
                    self.user_sentiment_history[author_id].pop(0)
                
                # Store current sentiment for history
                self.user_sentiment_history[author_id].append(raw_score)
                
                # Use sentiment history to provide context
                # If a user has been clearly positive/negative recently, this slightly influences current analysis
                if len(self.user_sentiment_history[author_id]) >= 3:
                    history_avg = sum(self.user_sentiment_history[author_id][-3:]) / 3
                    # Slight influence from history (20% weight)
                    raw_score = raw_score * 0.8 + history_avg * 0.2
                
                # Convert continuous score to discrete categories
                if raw_score > 1.0:
                    final_score = 2  # VERY_POSITIVE
                elif raw_score > 0.2:
                    final_score = 1  # POSITIVE
                elif raw_score < -1.0:
                    final_score = -2  # VERY_NEGATIVE
                elif raw_score < -0.2:
                    final_score = -1  # NEGATIVE
                else:
                    final_score = 0  # NEUTRAL
                
                results.append({
                    'message_id': message['id'],
                    'sentiment_score': final_score,
                    'sentiment_label': sentiment_map[final_score],
                    'raw_score': raw_score
                })
            
            print(f"Sentiment analysis completed for {len(results)} messages")
            return results
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            traceback.print_exc()
            return [{'message_id': m['id'], 'sentiment_score': 0, 'sentiment_label': 'NEUTRAL', 'raw_score': 0} for m in messages_batch]

    async def scrape_channel(self, channel_id, channel_name, guild_id, max_messages=500):
        """Scrape up to max_messages from a channel and analyze sentiment."""
        message_count = 0
        filtered_count = 0
        last_message_id = None
        all_messages = []
        
        print(f"Scraping up to {max_messages} messages from channel {channel_name} (ID: {channel_id})")
        
        # Check channel access first
        if not await self.check_channel_access(channel_id):
            print("Cannot access this channel. Make sure your token has permission to view it.")
            print("You need to be a member of the server and have permission to view this channel.")
            return
        
        # Collect messages first
        while message_count < max_messages:
            messages = await self.get_messages(channel_id, limit=100, before=last_message_id)
            if not messages:
                break
                
            for message in messages:
                if message_count >= max_messages:
                    break
                    
                # Check if message meets filter criteria
                if not await self.should_include_message(message, guild_id):
                    filtered_count += 1
                    continue
                
                all_messages.append(message)
                message_count += 1
                
                if message_count % 100 == 0:
                    print(f"Collected {message_count} messages from {channel_name}")
            
            if message_count >= max_messages or len(messages) < 100:
                break
                
            last_message_id = messages[-1]['id']
            # Add a small delay to avoid rate limits
            await asyncio.sleep(1)
        
        print(f"Collected {message_count} messages from {channel_name}")
        
        if message_count == 0:
            print("No messages were collected. This could be due to:")
            print("1. The channel being empty")
            print("2. Your token not having permission to view messages in this channel")
            print("3. The channel ID being incorrect")
            return
        
        # Analyze sentiment for collected messages
        batch_size = 50  # Process in smaller batches
        sentiment_results = []
        
        for i in range(0, len(all_messages), batch_size):
            batch = all_messages[i:i+batch_size]
            batch_results = self.analyze_sentiment(batch)
            sentiment_results.extend(batch_results)
            print(f"Analyzed batch {i//batch_size + 1}/{(len(all_messages)+batch_size-1)//batch_size}")
        
        # Create a lookup for sentiment results
        sentiment_lookup = {result['message_id']: result for result in sentiment_results}
        
        # Process messages with sentiment analysis
        for message in all_messages:
            sentiment_info = sentiment_lookup.get(message['id'], 
                                              {'sentiment_score': 0, 'sentiment_label': 'NEUTRAL', 'raw_score': 0})
            
            message_data = {
                'channel_name': channel_name,
                'channel_id': channel_id,
                'message_id': message['id'],
                'author': f"{message['author']['username']}#{message['author']['discriminator']}",
                'author_id': message['author']['id'],
                'content': message['content'],
                'created_at': message['timestamp'],
                'edited_at': message.get('edited_timestamp'),
                'sentiment_score': sentiment_info['sentiment_score'],
                'sentiment_label': sentiment_info['sentiment_label'],
                'raw_sentiment_score': sentiment_info.get('raw_score', 0)
            }
            self.messages_data.append(message_data)
        
        print(f"Finished processing {channel_name} - Total messages: {message_count} (Filtered: {filtered_count})")
        
        # Calculate overall sentiment statistics
        if self.messages_data:
            df = pd.DataFrame(self.messages_data)
            sentiment_counts = df['sentiment_label'].value_counts()
            sentiment_avg = df['sentiment_score'].mean()
            
            print("\nSentiment Analysis Results:")
            print(f"Average Sentiment Score: {sentiment_avg:.2f}")
            print("Sentiment Distribution:")
            for label, count in sentiment_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {label}: {count} messages ({percentage:.1f}%)")

    async def scrape(self):
        """Main scraping function."""
        try:
            guilds = await self.get_guilds()
            
            # Find Bittensor guild
            bittensor_guild = None
            for guild in guilds:
                if 'bittensor' in guild['name'].lower():
                    bittensor_guild = guild
                    break
            
            if not bittensor_guild:
                print("Could not find Bittensor guild. Are you a member?")
                return
                
            print(f"Found guild: {bittensor_guild['name']}")
            
            # Get roles and let user select which ones to filter by
            roles = await self.get_guild_roles(bittensor_guild['id'])
            if not roles:
                print("No roles found in the server.")
                return
                
            self.target_roles = self.display_and_select_roles(roles)
            if not self.target_roles:
                print("No roles selected. Exiting...")
                return
                
            # Get channels
            channels = await self.get_channels(bittensor_guild['id'])
            text_channels = [c for c in channels if c['type'] == 0]  # 0 is text channel
            
            for channel in text_channels:
                try:
                    print(f"Scraping channel: {channel['name']}")
                    await self.scrape_channel(channel['id'], channel['name'], bittensor_guild['id'])
                except Exception as e:
                    print(f"Error in channel {channel['name']}: {str(e)}")
            
            # Final save
            self.save_to_csv()
            print("Scraping completed!")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            
    def save_to_csv(self):
        """Save the collected messages to a CSV file."""
        if not self.messages_data:
            print("No messages were collected to save.")
            return
            
        df = pd.DataFrame(self.messages_data)
        output_dir = 'scraped_data'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{output_dir}/discord_messages_{timestamp}.csv'
        
        df.to_csv(filename, index=False)
        print(f"Saved {len(self.messages_data)} messages to {filename}")
        
        # Also save sentiment analysis summary
        sentiment_summary = df['sentiment_label'].value_counts().reset_index()
        sentiment_summary.columns = ['sentiment', 'count']
        sentiment_summary['percentage'] = (sentiment_summary['count'] / len(df)) * 100
        
        # Calculate additional stats
        sentiment_by_user = df.groupby('author')['sentiment_score'].agg(['mean', 'count']).reset_index()
        sentiment_by_user.columns = ['author', 'avg_sentiment', 'message_count']
        sentiment_by_user = sentiment_by_user.sort_values('avg_sentiment', ascending=False)
        
        # Identify most positive and negative users
        top_positive_users = sentiment_by_user[sentiment_by_user['message_count'] >= 3].head(5)
        top_negative_users = sentiment_by_user[sentiment_by_user['message_count'] >= 3].tail(5)
        
        # Save detailed sentiment summary
        summary_filename = f'{output_dir}/sentiment_summary_{timestamp}.csv'
        sentiment_summary.to_csv(summary_filename, index=False)
        
        # Save user sentiment stats
        user_sentiment_filename = f'{output_dir}/user_sentiment_{timestamp}.csv'
        sentiment_by_user.to_csv(user_sentiment_filename, index=False)
        
        print(f"Saved sentiment summary to {summary_filename}")
        print(f"Saved user sentiment analysis to {user_sentiment_filename}")
        
        # Print sentiment analysis summary
        print("\nSentiment Analysis Summary:")
        print("---------------------------")
        for index, row in sentiment_summary.iterrows():
            print(f"{row['sentiment']}: {row['count']} messages ({row['percentage']:.1f}%)")
            
        print("\nTop 5 Most Positive Users:")
        for index, row in top_positive_users.iterrows():
            print(f"{row['author']}: {row['avg_sentiment']:.2f} (from {row['message_count']} messages)")
            
        if not top_negative_users.empty:
            print("\nTop 5 Most Negative Users:")
            for index, row in top_negative_users.iterrows():
                print(f"{row['author']}: {row['avg_sentiment']:.2f} (from {row['message_count']} messages)")
        
async def main():
    # Check if help is requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        display_help()
        return
    
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        print("Discord token not found. Please set your Discord token in the .env file")
        display_help()
        sys.exit(1)
        
    # Print token info (safely)
    if token:
        print(f"Using token: {token[:5]}...{token[-4:]} (length: {len(token)})")
    
    scraper = BittensorDiscordScraper(token)
    
    # Validate token first
    if not await scraper.validate_token():
        print("Token validation failed. Please check your Discord token.")
        display_help()
        sys.exit(1)
    
    try:
        # Use the specific subnet channel ID from the request
        SUBNET_CHANNEL_ID = "1161764867166961704"
        
        print(f"Starting to scrape subnet channel (ID: {SUBNET_CHANNEL_ID})")
        
        # Scrape the specified channel directly with 500 message limit
        await scraper.scrape_channel(SUBNET_CHANNEL_ID, "subnet-channel", "bittensor-guild", max_messages=500)
        
        # Save the final results
        scraper.save_to_csv()
        print("Scraping and sentiment analysis completed!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    asyncio.run(main())
