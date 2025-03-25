#!/usr/bin/env python3
"""
reset_and_scrape.py

This script:
1. Resets the database (deletes all messages) after making a backup
2. Scrapes exactly 100 messages from each subnet
3. Runs sentiment analysis on each subnet
"""

import os
import sqlite3
import logging
import asyncio
import time
from dotenv import load_dotenv
from enhanced_scraper import EnhancedDiscordScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ResetAndScrape')

# Load environment variables
load_dotenv()

# Database path - must match the one in enhanced_scraper.py
DATABASE_PATH = 'discord_messages.db'

async def reset_database():
    """Reset the database by deleting all messages."""
    try:
        # Check if database exists
        if not os.path.exists(DATABASE_PATH):
            logger.info(f"Database {DATABASE_PATH} does not exist yet. Nothing to reset.")
            return True
        
        # Back up the database first
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = "backups"
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"discord_messages_{timestamp}.db")
        
        logger.info(f"Backing up database to {backup_path}")
        with open(DATABASE_PATH, 'rb') as source:
            with open(backup_path, 'wb') as dest:
                dest.write(source.read())
        
        # Connect to the database and delete all messages
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check how many messages we're deleting
        cursor.execute("SELECT COUNT(*) FROM messages")
        message_count = cursor.fetchone()[0]
        logger.info(f"Deleting {message_count} messages from the database")
        
        # Delete all messages
        cursor.execute("DELETE FROM messages")
        conn.commit()
        
        # Verify deletion
        cursor.execute("SELECT COUNT(*) FROM messages")
        new_count = cursor.fetchone()[0]
        logger.info(f"Database now contains {new_count} messages")
        
        conn.close()
        logger.info("Database reset complete")
        return True
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        return False

async def main():
    """Main function to orchestrate the reset, scraping, and analysis."""
    # Define the subnets to process
    subnets = [1, 3, 4, 5, 8, 9, 11, 16, 19, 21]
    
    # Step 1: Reset the database
    if not await reset_database():
        logger.error("Database reset failed. Aborting.")
        return
    
    # Step 2: Create an instance of the scraper
    scraper = EnhancedDiscordScraper()
    scraper.initialize_database()
    
    # Step 3: Scrape messages from each subnet (limited to 100 per subnet)
    for subnet in subnets:
        logger.info(f"Scraping subnet {subnet}")
        
        # Get the channel ID for this subnet
        channel_id = scraper.subnet_ids.get(str(subnet))
        if not channel_id:
            logger.warning(f"No channel ID found for subnet {subnet}. Skipping.")
            continue
        
        # Scrape messages from this subnet
        try:
            # Get the last 100 messages from the channel
            messages = await scraper.get_messages(channel_id, limit=100)
            
            if not messages:
                logger.warning(f"No messages found for subnet {subnet}")
                continue
                
            logger.info(f"Retrieved {len(messages)} messages for subnet {subnet}")
            
            # Store messages in the database
            scraper.store_messages(messages, subnet)
            
            logger.info(f"Successfully scraped and stored {len(messages)} messages for subnet {subnet}")
        except Exception as e:
            logger.error(f"Error scraping subnet {subnet}: {e}")
    
    # Step 4: Run sentiment analysis on all unanalyzed messages
    logger.info("Processing unanalyzed messages...")
    await scraper.process_unanalyzed_messages()
    
    # Step 5: Generate sentiment reports
    logger.info("Generating sentiment reports...")
    scraper.generate_sentiment_report()
    
    # Step 6: For each subnet, run recent sentiment analysis
    for subnet in subnets:
        logger.info(f"Analyzing recent sentiment for subnet {subnet}")
        try:
            overall_sentiment, message_sentiments = await scraper.analyze_recent_messages(limit=100, subnet_num=subnet)
            
            logger.info(f"Subnet {subnet} overall sentiment: {overall_sentiment}")
            logger.info(f"Analyzed {len(message_sentiments)} messages")
            
            # Print distribution of sentiments
            positive = sum(1 for s in message_sentiments if s > 0)
            neutral = sum(1 for s in message_sentiments if s == 0)
            negative = sum(1 for s in message_sentiments if s < 0)
            
            logger.info(f"Sentiment distribution for subnet {subnet}:")
            logger.info(f"  Positive: {positive} ({positive/len(message_sentiments)*100:.2f}%)")
            logger.info(f"  Neutral: {neutral} ({neutral/len(message_sentiments)*100:.2f}%)")
            logger.info(f"  Negative: {negative} ({negative/len(message_sentiments)*100:.2f}%)")
        except Exception as e:
            logger.error(f"Error analyzing sentiment for subnet {subnet}: {e}")
    
    # Step 7: Run combined sentiment analysis
    logger.info("Analyzing recent sentiment for all subnets combined")
    try:
        overall_sentiment, message_sentiments = await scraper.analyze_recent_messages(limit=100)
        
        logger.info(f"Overall sentiment across all subnets: {overall_sentiment}")
        logger.info(f"Analyzed {len(message_sentiments)} messages")
        
        # Print distribution of sentiments
        positive = sum(1 for s in message_sentiments if s > 0)
        neutral = sum(1 for s in message_sentiments if s == 0)
        negative = sum(1 for s in message_sentiments if s < 0)
        
        logger.info(f"Overall sentiment distribution:")
        logger.info(f"  Positive: {positive} ({positive/len(message_sentiments)*100:.2f}%)")
        logger.info(f"  Neutral: {neutral} ({neutral/len(message_sentiments)*100:.2f}%)")
        logger.info(f"  Negative: {negative} ({negative/len(message_sentiments)*100:.2f}%)")
    except Exception as e:
        logger.error(f"Error analyzing overall sentiment: {e}")
    
    logger.info("Reset, scraping, and analysis complete!")

if __name__ == "__main__":
    asyncio.run(main())
