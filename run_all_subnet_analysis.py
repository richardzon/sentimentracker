#!/usr/bin/env python3
"""
Script to analyze sentiment for all subnets using the EnhancedDiscordScraper.
This populates the sentiment dashboard with data from all subnets.
"""

import os
import sys
import json
import asyncio
import logging
import shutil
from datetime import datetime
from dotenv import load_dotenv
from enhanced_scraper import EnhancedDiscordScraper, OUTPUT_DIR
from sentiment_tier_manager import SentimentTierManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("all_subnet_analysis.log"), logging.StreamHandler()]
)
logger = logging.getLogger("all_subnet_analyzer")

# Constants
DASHBOARD_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "subnet-sentiment-dashboard", "public", "data")
ALL_SUBNETS_FILE = os.path.join(DASHBOARD_DATA_DIR, "all_subnets_latest.json")
SUBNET_SENTIMENT_FILE = os.path.join(DASHBOARD_DATA_DIR, "subnet_sentiment.json")

# Make sure output directories exist
os.makedirs(DASHBOARD_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DASHBOARD_DATA_DIR, "subnet_analysis"), exist_ok=True)
os.makedirs(os.path.join(DASHBOARD_DATA_DIR, "history"), exist_ok=True)

async def analyze_subnet(scraper, subnet_num, channel_id):
    """Analyze a single subnet and return its data"""
    logger.info(f"Analyzing subnet {subnet_num}...")
    
    try:
        # Get messages from the subnet
        messages = await scraper.scrape_subnet(subnet_num, channel_id)
        
        if not messages:
            logger.warning(f"No messages found for subnet {subnet_num}")
            return None
        
        # Format messages for analysis
        formatted_messages = await scraper.format_messages_for_analysis(messages, subnet_num)
        
        # Analyze sentiment using Qwen model (with TextBlob as fallback for proper technical detection)
        sentiment_score = await scraper.analyze_subnet_sentiment_batch(formatted_messages, subnet_num)
        
        # Calculate sentiment distribution
        positive_count = sum(1 for msg in messages if msg.get('sentiment', 0) > 0.1)
        negative_count = sum(1 for msg in messages if msg.get('sentiment', 0) < -0.1)
        neutral_count = len(messages) - positive_count - negative_count
        
        # Calculate percentages
        total = len(messages)
        sentiment_distribution = {
            "positive": round((positive_count / total) * 100) if total > 0 else 0,
            "negative": round((negative_count / total) * 100) if total > 0 else 0,
            "neutral": round((neutral_count / total) * 100) if total > 0 else 0
        }
        
        # Get the detailed analysis data
        analysis_path = os.path.join(OUTPUT_DIR, "subnet_analysis", f"subnet_{subnet_num}_analysis.json")
        detailed_analysis = {}
        
        if os.path.exists(analysis_path):
            try:
                with open(analysis_path, 'r') as f:
                    detailed_analysis = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Could not parse analysis file for subnet {subnet_num}")
        
        # Calculate technical question percentage
        technical_percentage = detailed_analysis.get("technical_percentage", 
                                                    scraper.calculate_technical_percentage(messages))
        
        # Extract key topics
        topics = detailed_analysis.get("key_topics", 
                                      scraper.extract_key_topics(messages))
        
        # Check for subnet rating based on const's messages (time-based tier system)
        tier = scraper.check_subnet_rating(subnet_num)
        logger.info(f"Subnet {subnet_num} tier result: {tier}")
        
        # Ensure tier has a default value if none was assigned
        if tier is None:
            tier = "standard"
        
        logger.info(f"Final tier for subnet {subnet_num}: {tier}")
        
        # Format top recent messages
        recent_messages = []
        for msg in messages[:10]:  # Just use top 10 messages
            if 'content' in msg and 'author' in msg:
                recent_messages.append({
                    "content": msg['content'][:100] + ("..." if len(msg['content']) > 100 else ""),
                    "author": msg['author'],
                    "sentiment": msg.get('sentiment', 0)
                })
        
        # Create the subnet data structure
        subnet_data = {
            "subnet": subnet_num,
            "average_sentiment": round(sentiment_score, 2),
            "message_count": len(messages),
            "sentiment_distribution": detailed_analysis.get("sentiment_breakdown", sentiment_distribution),
            "technical_percentage": technical_percentage,
            "key_topics": topics,
            "tier": tier,  # Add the const message freshness tier
            "const_rating": tier is not None,  # Flag if const has rated this subnet
            "messages": recent_messages,
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(f"Subnet {subnet_num} data structure created with tier: {tier}")
        logger.debug(f"Final subnet data keys: {list(subnet_data.keys())}")
        
        # Save individual subnet file for dashboard
        subnet_file = os.path.join(DASHBOARD_DATA_DIR, f"subnet_{subnet_num}_latest.json")
        with open(subnet_file, 'w') as f:
            json_str = json.dumps(subnet_data, indent=2)
            f.write(json_str)
            logger.debug(f"Wrote {len(json_str)} bytes to {subnet_file}")
        
        # Verify the file was written correctly
        if os.path.exists(subnet_file):
            with open(subnet_file, 'r') as f:
                verification = json.load(f)
                logger.debug(f"Verification - tier in file: {'tier' in verification}")
                
        # Save simple file for legacy code
        simple_file = os.path.join(DASHBOARD_DATA_DIR, f"subnet_{subnet_num}.json")
        simple_data = {
            "subnet": subnet_num,
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
            "avg_sentiment": round(sentiment_score, 2),
            "updated_at": datetime.now().isoformat(),
            # Add tier information to the simple file format as well
            "tier": tier,
            "const_rating": tier is not None
        }
        with open(simple_file, 'w') as f:
            json.dump(simple_data, f, indent=2)
        
        # Ensure the analysis directory exists
        dashboard_analysis_dir = os.path.join(DASHBOARD_DATA_DIR, "analysis")
        os.makedirs(dashboard_analysis_dir, exist_ok=True)
        dashboard_analysis_path = os.path.join(dashboard_analysis_dir, f"subnet_{subnet_num}_analysis.json")
        
        # If analysis exists, copy it to the dashboard
        if os.path.exists(analysis_path):
            shutil.copy2(analysis_path, dashboard_analysis_path)
            logger.info(f"Copied analysis to dashboard: {dashboard_analysis_path}")
        
        # Save to history
        history_dir = os.path.join(DASHBOARD_DATA_DIR, "history", datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(history_dir, exist_ok=True)
        history_file = os.path.join(history_dir, f"subnet_{subnet_num}_{datetime.now().strftime('%H%M%S')}.json")
        with open(history_file, 'w') as f:
            json.dump(subnet_data, f, indent=2)
            
        logger.info(f"Analysis complete for subnet {subnet_num}: {round(sentiment_score, 2)} avg sentiment, {len(messages)} messages, tier: {tier}")
        
        return subnet_data
        
    except Exception as e:
        logger.exception(f"Error analyzing subnet {subnet_num}: {str(e)}")
        return None

async def main():
    logger.info("Starting analysis of all subnets...")
    
    # Load environment variables
    load_dotenv()
    discord_token = os.getenv('DISCORD_USER_TOKEN')
    
    if not discord_token:
        logger.error("DISCORD_USER_TOKEN environment variable not set")
        sys.exit(1)
        
    # Initialize scraper
    scraper = EnhancedDiscordScraper()
    
    # Initialize tier manager
    tier_manager = SentimentTierManager()
    
    # Get subnet mappings
    subnet_mapping = scraper.load_subnet_ids()
    
    if not subnet_mapping:
        logger.error("Failed to load subnet mappings")
        sys.exit(1)
        
    # Filter subnets - skip subnet 0 which is general
    filtered_subnets = {subnet_id: channel_id for subnet_id, channel_id in subnet_mapping.items() 
                      if int(subnet_id) != 0}
    
    logger.info(f"Found {len(filtered_subnets)} subnets for analysis")
    
    # Reset lists
    all_subnet_data = []
    
    # Process in smaller batches to avoid rate limiting (5 subnets at a time)
    batch_size = 5
    batches = [(a, b) for a, b in zip(
        list(filtered_subnets.keys())[::batch_size],
        [list(filtered_subnets.keys())[i:i+batch_size] for i in range(0, len(filtered_subnets), batch_size)]
    )]
    
    for batch_num, subnet_batch in enumerate(batches, 1):
        logger.info(f"Processing batch {batch_num}/{len(batches)}")
        
        # Process batch
        batch_data = []
        for subnet_num in subnet_batch[1]:
            channel_id = filtered_subnets[subnet_num]
            subnet_data = await analyze_subnet(scraper, subnet_num, channel_id)
            
            if subnet_data:
                all_subnet_data.append(subnet_data)
                batch_data.append(subnet_data)
                
        # Wait between batches to avoid rate limiting
        if batch_num < len(batches):
            logger.info("Waiting 5 seconds before next batch...")
            await asyncio.sleep(5)
    
    # Sort by subnet number
    all_subnet_data.sort(key=lambda x: int(x['subnet']))
    
    # Make sure all subnets have their tier information properly set
    for subnet_data in all_subnet_data:
        if 'tier' not in subnet_data or subnet_data['tier'] is None:
            subnet_data['tier'] = 'standard'
            logger.warning(f"Added missing tier for subnet {subnet_data['subnet']}")
    
    logger.info(f"Total subnets analyzed: {len(all_subnet_data)}")
    
    # Save all subnets data to a single file
    if all_subnet_data:
        logger.info(f"Writing data for {len(all_subnet_data)} subnets to {ALL_SUBNETS_FILE}")
        with open(ALL_SUBNETS_FILE, 'w') as f:
            json.dump(all_subnet_data, f, indent=2)
            
        # Also update the subnet_sentiment.json file used by the dashboard
        logger.info(f"Updating dashboard file at {SUBNET_SENTIMENT_FILE}")
        with open(SUBNET_SENTIMENT_FILE, 'w') as f:
            json.dump(all_subnet_data, f, indent=2)
            
        # Verify tier information was saved correctly
        try:
            with open(ALL_SUBNETS_FILE, 'r') as f:
                verification = json.load(f)
                logger.info(f"Verification: Successfully loaded {len(verification)} subnets from {ALL_SUBNETS_FILE}")
                for subnet in verification:
                    if 'tier' not in subnet:
                        logger.error(f"VERIFICATION FAILED: Subnet {subnet.get('subnet', 'unknown')} missing tier information")
                    else:
                        logger.debug(f"Verification successful: Subnet {subnet.get('subnet')} has tier {subnet.get('tier')}")
        except Exception as e:
            logger.error(f"Error during verification: {e}")
    else:
        logger.error("No subnet data was collected to write to all_subnets_latest.json")
    
    logger.info(f"Analysis completed for {len(all_subnet_data)} subnets.")
    logger.info(f"Data saved to {ALL_SUBNETS_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
