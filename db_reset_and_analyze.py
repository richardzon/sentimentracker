#!/usr/bin/env python3
"""
db_reset_and_analyze.py

A direct script to:
1. Delete all data from the Discord messages database
2. Run the scraper to get exactly 100 messages per subnet
3. Run sentiment analysis on each subnet
"""

import os
import sqlite3
import logging
import sys
import subprocess
import time
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DBResetAndAnalyze')

# Load environment variables
load_dotenv()

# Use the exact same database path as in enhanced_scraper.py
DATABASE_PATH = 'discord_messages.db'

def reset_database():
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
        
        # Connect to the database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check how many messages we're deleting
        try:
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
        except sqlite3.OperationalError:
            logger.info("Table 'messages' does not exist yet. Nothing to delete.")
        
        conn.close()
        logger.info("Database reset complete")
        return True
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        return False

def run_enhanced_scraper():
    """Run the enhanced scraper for all subnets."""
    logger.info("Running enhanced scraper...")
    
    result = subprocess.run(
        ["python", "enhanced_scraper.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Enhanced scraper failed with exit code {result.returncode}")
        logger.error(result.stderr)
        return False
    
    logger.info("Enhanced scraper completed successfully")
    return True

def run_analyze_recent():
    """Run sentiment analysis on the most recent messages."""
    logger.info("Running sentiment analysis on recent messages...")
    
    # Define the subnets
    subnets = [1, 3, 4, 5, 8, 9, 11, 16, 19, 21]
    
    # Analyze each subnet
    for subnet in subnets:
        logger.info(f"Analyzing sentiment for subnet {subnet}...")
        
        result = subprocess.run(
            ["python", "analyze_recent.py", "--limit", "100", "--subnet", str(subnet)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Sentiment analysis for subnet {subnet} failed")
            logger.error(result.stderr)
        else:
            print(f"\n----- SENTIMENT ANALYSIS FOR SUBNET {subnet} -----")
            print(result.stdout)
            print(f"----- END SUBNET {subnet} -----\n")
    
    # Run analysis for all subnets combined
    logger.info("Analyzing sentiment for all subnets combined...")
    
    result = subprocess.run(
        ["python", "analyze_recent.py", "--limit", "100"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error("Combined sentiment analysis failed")
        logger.error(result.stderr)
    else:
        print("\n----- COMBINED SENTIMENT ANALYSIS FOR ALL SUBNETS -----")
        print(result.stdout)
        print("----- END COMBINED ANALYSIS -----\n")
    
    return True

def main():
    """Main function to reset database, run scraper, and analyze sentiment."""
    # Step 1: Reset the database
    if not reset_database():
        logger.error("Failed to reset database. Aborting.")
        sys.exit(1)
    
    # Step 2: Run the enhanced scraper to populate the database
    if not run_enhanced_scraper():
        logger.error("Failed to run enhanced scraper. Aborting.")
        sys.exit(1)
    
    # Step 3: Run sentiment analysis on the most recent messages
    if not run_analyze_recent():
        logger.error("Failed to run sentiment analysis.")
        sys.exit(1)
    
    logger.info("Database reset, scraping, and sentiment analysis completed successfully.")

if __name__ == "__main__":
    main()
