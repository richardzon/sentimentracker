#!/usr/bin/env python3
"""
reset_scrape_analyze.py

Script to:
1. Reset the database (delete all messages)
2. Scrape 100 messages from each subnet
3. Run sentiment analysis on each subnet
"""

import os
import sqlite3
import logging
import asyncio
import subprocess
import time
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ResetScrapeAnalyze')

# Database path (hardcoded to match enhanced_scraper.py)
DATABASE_PATH = 'discord_messages.db'

def reset_database():
    """Reset the database by deleting all messages."""
    try:
        # Check if database exists first
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

def run_command(cmd, description):
    """Run a command and log the output."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        logger.info(f"Command succeeded: {description}")
        logger.debug(f"Output: {result.stdout}")
        
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")
            
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {description}")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False

def main():
    """Main function to orchestrate the reset, scraping, and analysis."""
    # Define the subnets to process
    subnets = [1, 3, 4, 5, 8, 9, 11, 16, 19, 21]
    
    # Step 1: Reset the database
    if not reset_database():
        logger.error("Database reset failed. Aborting.")
        return
    
    # Step 2: Scrape messages from each subnet (limited to 100 per subnet)
    for subnet in subnets:
        # Run scraper for this subnet
        success = run_command(
            ["python", "enhanced_scraper.py", "--limited", "--subnet", str(subnet)],
            f"Scraping subnet {subnet}"
        )
        
        if not success:
            logger.warning(f"Scraping subnet {subnet} may have had issues")
    
    # Optional: Sleep to ensure database operations are complete
    logger.info("Waiting for database operations to complete...")
    time.sleep(2)
    
    # Step 3: Run sentiment analysis on the scraped messages
    # First, check how many messages we have for each subnet
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    for subnet in subnets:
        try:
            cursor.execute("SELECT COUNT(*) FROM messages WHERE subnet = ?", (subnet,))
            count = cursor.fetchone()[0]
            logger.info(f"Subnet {subnet} has {count} messages")
            
            # Only analyze if we have messages
            if count > 0:
                # Now run the analysis for this subnet
                success = run_command(
                    ["python", "analyze_recent.py", "--limit", "100", "--subnet", str(subnet)],
                    f"Analyzing sentiment for subnet {subnet}"
                )
                
                if not success:
                    logger.warning(f"Sentiment analysis for subnet {subnet} may have had issues")
        except Exception as e:
            logger.error(f"Error checking message count for subnet {subnet}: {e}")
    
    # Run analysis for all subnets combined
    run_command(
        ["python", "analyze_recent.py", "--limit", "100"],
        "Analyzing sentiment for all subnets combined"
    )
    
    conn.close()
    logger.info("Reset, scraping, and analysis complete!")

if __name__ == "__main__":
    main()
