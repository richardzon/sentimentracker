#!/usr/bin/env python3

import os
import json
import logging
import sqlite3
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SentimentTierManager')

class SentimentTierManager:
    """
    Class to manage the sentiment tier system for "const" user messages.
    This provides tier ratings (emerald, golden, silver, bronze) based on message freshness.
    """
    
    def __init__(self, db_path="discord_messages.db", dashboard_dir="subnet-sentiment-dashboard/public/data"):
        """Initialize the tier manager with database and output directory paths."""
        self.db_path = db_path
        self.dashboard_dir = dashboard_dir
        
        # Create connection to the database
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # Ensure dashboard directory exists
        os.makedirs(self.dashboard_dir, exist_ok=True)
        os.makedirs(os.path.join(self.dashboard_dir, "tiers"), exist_ok=True)
        
        logger.info(f"SentimentTierManager initialized with database: {db_path}")
    
    def get_subnet_tier(self, subnet_num):
        """
        Get the tier rating for a specific subnet based on 'const' user messages.
        
        Tier system:
        - Emerald: Messages from 'const' within the last hour
        - Golden: Messages from 'const' within the last 6 hours
        - Silver: Messages from 'const' within the last 24 hours
        - Bronze: Messages from 'const' within the last 48 hours
        """
        # Get the current time (always use timezone-naive)
        now = datetime.now()
        
        # First check if there are any messages from the const user
        self.cursor.execute("""
            SELECT COUNT(*) 
            FROM messages 
            WHERE subnet_num = ? 
              AND author LIKE 'const%'
        """, (subnet_num,))
        
        count = self.cursor.fetchone()[0]
        if count == 0:
            logger.debug(f"No messages from 'const' found for subnet {subnet_num}")
            return None
        
        logger.debug(f"Found {count} messages from 'const' for subnet {subnet_num}")
        
        # Query for the most recent message from 'const', regardless of sentiment
        self.cursor.execute("""
            SELECT timestamp, content 
            FROM messages 
            WHERE subnet_num = ? 
              AND author LIKE 'const%' 
            ORDER BY timestamp DESC
            LIMIT 1
        """, (subnet_num,))
        
        result = self.cursor.fetchone()
        
        if not result:
            logger.warning(f"Database query for 'const' messages returned no results despite count={count}")
            return None
        
        # Get the timestamp of the most recent message, regardless of sentiment
        timestamp, content = result["timestamp"], result["content"]
        logger.debug(f"Processing timestamp format: {timestamp!r}")
        
        # Parse the timestamp and calculate age
        try:
            # Standardized timestamp handling with proper timezone handling
            message_time = None
            try:
                # Handle ISO format with timezone
                if 'Z' in timestamp:
                    # Convert UTC 'Z' format to timezone-naive by removing the Z and interpreting as UTC
                    timestamp_naive = timestamp.replace('Z', '')
                    message_time = datetime.fromisoformat(timestamp_naive)
                elif '+' in timestamp or '-' in timestamp and 'T' in timestamp:
                    # Handle timestamps with explicit timezone offsets
                    # Parse with timezone, then convert to naive by removing timezone info
                    message_time = datetime.fromisoformat(timestamp).replace(tzinfo=None)
                else:
                    # Handle timezone-naive ISO format
                    message_time = datetime.fromisoformat(timestamp)
            except ValueError:
                try:
                    # Try common formats
                    if 'T' in timestamp:
                        message_time = datetime.strptime(timestamp.split('.')[0], '%Y-%m-%dT%H:%M:%S')
                    else:
                        message_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    # Last resort: Just use current time minus a default age
                    logger.warning(f"Could not parse timestamp: {timestamp!r}, using current time minus 1 day")
                    message_time = now - timedelta(days=1)  # Default to 1 day old
            
            # Ensure message_time is timezone-naive for comparison with now (which is also naive)
            if hasattr(message_time, 'tzinfo') and message_time.tzinfo is not None:
                message_time = message_time.replace(tzinfo=None)
                
            message_age = now - message_time
            
            # Log the message age for debugging
            logger.debug(f"Message from 'const' is {message_age} old (timestamp: {timestamp})")
            
            # Assign tier based on age - using "golden" instead of "gold" to match dashboard expectations
            if message_age <= timedelta(hours=1):
                logger.info(f"Assigning EMERALD tier to subnet {subnet_num}: message age = {message_age}")
                return "emerald"
            elif message_age <= timedelta(hours=6):
                logger.info(f"Assigning GOLDEN tier to subnet {subnet_num}: message age = {message_age}")
                return "golden"
            elif message_age <= timedelta(hours=24):
                logger.info(f"Assigning SILVER tier to subnet {subnet_num}: message age = {message_age}")
                return "silver"
            elif message_age <= timedelta(hours=48):
                logger.info(f"Assigning BRONZE tier to subnet {subnet_num}: message age = {message_age}")
                return "bronze"
            else:
                logger.info(f"Assigning STANDARD tier to subnet {subnet_num}: message age = {message_age}")
                return "standard"  # Older than 48 hours
        except Exception as e:
            logger.error(f"Error parsing timestamp: {str(e)}")
            return None
    
    def update_all_subnet_tiers(self, subnet_list=None):
        """
        Update tier data for all subnets or a specific list of subnets.
        
        Saves tier information to a dedicated tier JSON file for each subnet
        and also updates the tier information in the main subnet JSON files.
        """
        # If no subnet list is provided, get all subnets from the database
        if subnet_list is None:
            self.cursor.execute("SELECT DISTINCT subnet_num FROM messages")
            subnet_list = [row[0] for row in self.cursor.fetchall()]
        
        tier_data = {}
        success_count = 0
        
        for subnet_num in subnet_list:
            try:
                # Get the tier for this subnet - calling the correct method
                tier = self.get_subnet_tier(subnet_num)
                
                # Skip if no tier
                if tier is None:
                    continue
                
                # Add to the overall tier data
                tier_data[subnet_num] = {
                    "tier": tier,
                    "const_rating": True,
                    "updated_at": datetime.now().isoformat()
                }
                
                # Save individual tier file
                tier_file = os.path.join(self.dashboard_dir, "tiers", f"subnet_{subnet_num}_tier.json")
                with open(tier_file, 'w') as f:
                    json.dump(tier_data[subnet_num], f, indent=2)
                
                # Also update the main subnet file if it exists
                latest_file = os.path.join(self.dashboard_dir, f"subnet_{subnet_num}_latest.json")
                if os.path.exists(latest_file):
                    try:
                        with open(latest_file, 'r') as f:
                            subnet_data = json.load(f)
                        
                        # Add tier information in the format the dashboard expects
                        subnet_data["tier"] = tier
                        subnet_data["const_rating"] = True
                        
                        # IMPORTANT: Add is_golden property that the dashboard looks for
                        subnet_data["is_golden"] = (tier == "golden" or tier == "emerald")
                        
                        # Save the updated file
                        with open(latest_file, 'w') as f:
                            json.dump(subnet_data, f, indent=2)
                        
                        logger.debug(f"Updated tier information in main subnet file: {latest_file}")
                    except Exception as e:
                        logger.error(f"Error updating main subnet file: {e}")
                
                # Update the simple subnet file
                simple_file = os.path.join(self.dashboard_dir, f"subnet_{subnet_num}.json")
                if os.path.exists(simple_file):
                    try:
                        with open(simple_file, 'r') as f:
                            simple_data = json.load(f)
                        
                        # Add tier information in the format the dashboard expects
                        simple_data["tier"] = tier
                        simple_data["const_rating"] = True
                        
                        # IMPORTANT: Add is_golden property that the dashboard looks for
                        simple_data["is_golden"] = (tier == "golden" or tier == "emerald")
                        
                        # Save the updated file
                        with open(simple_file, 'w') as f:
                            json.dump(simple_data, f, indent=2)
                        
                        logger.debug(f"Updated tier information in simple subnet file: {simple_file}")
                    except Exception as e:
                        logger.error(f"Error updating simple subnet file: {e}")
                
                # Also update the all_subnets_latest.json file as this is what the dashboard loads
                all_subnets_file = os.path.join(self.dashboard_dir, "all_subnets_latest.json")
                if os.path.exists(all_subnets_file):
                    try:
                        # IMPORTANT: Read the entire file contents first
                        with open(all_subnets_file, 'r') as f:
                            all_subnets_data = json.load(f)
                        
                        # If the file is empty or corrupt, initialize it as an empty dict
                        if not isinstance(all_subnets_data, dict):
                            logger.warning(f"all_subnets_latest.json data is not a dictionary, initializing fresh")
                            all_subnets_data = {}
                        
                        # Add tier information to the specific subnet in the all_subnets_latest.json file
                        if str(subnet_num) in all_subnets_data:
                            # Only update the tier fields, preserve all other fields
                            all_subnets_data[str(subnet_num)]["tier"] = tier
                            all_subnets_data[str(subnet_num)]["const_rating"] = True
                            all_subnets_data[str(subnet_num)]["is_golden"] = (tier == "golden" or tier == "emerald")
                            
                            # Save the updated file with atomic write
                            temp_file = all_subnets_file + ".tmp"
                            with open(temp_file, 'w') as f:
                                json.dump(all_subnets_data, f, indent=2)
                            
                            # Only replace the original file if the write was successful
                            os.replace(temp_file, all_subnets_file)
                            
                            logger.debug(f"Updated tier information in all_subnets_latest.json for subnet {subnet_num}")
                    except Exception as e:
                        logger.error(f"Error updating all_subnets_latest.json: {e}")
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing subnet {subnet_num}: {e}")
        
        # Generate CSS classes for the dashboard
        self._generate_tier_css()
        
        # Save the overall tier data
        all_tiers_file = os.path.join(self.dashboard_dir, "tiers", "all_subnet_tiers.json")
        with open(all_tiers_file, 'w') as f:
            json.dump(tier_data, f, indent=2)
        
        logger.info(f"Updated tier information for {success_count} subnets")
        return tier_data
    
    def _generate_tier_css(self):
        """Generate CSS classes for tier visualization in the dashboard"""
        css_content = """
/* Sentiment tier system CSS */
.tier-emerald {
    background-color: #50C878;
    color: white;
    border-radius: 4px;
    padding: 3px 8px;
    font-weight: bold;
}
.tier-golden {
    background-color: #FFD700;
    color: black;
    border-radius: 4px;
    padding: 3px 8px;
    font-weight: bold;
}
.tier-silver {
    background-color: #C0C0C0;
    color: black;
    border-radius: 4px;
    padding: 3px 8px;
    font-weight: bold;
}
.tier-bronze {
    background-color: #CD7F32;
    color: white;
    border-radius: 4px;
    padding: 3px 8px;
    font-weight: bold;
}
.tier-standard {
    background-color: #888888;
    color: white;
    border-radius: 4px;
    padding: 3px 8px;
    font-weight: bold;
}
"""
        # Save CSS file to the dashboard styles directory
        css_dir = os.path.join(os.path.dirname(self.dashboard_dir), "styles")
        os.makedirs(css_dir, exist_ok=True)
        
        css_file = os.path.join(css_dir, "tier-styles.css")
        with open(css_file, 'w') as f:
            f.write(css_content)
        
        logger.debug(f"Generated tier CSS styles file at: {css_file}")

if __name__ == "__main__":
    # Run the tier manager as a standalone script
    tier_manager = SentimentTierManager()
    tier_data = tier_manager.update_all_subnet_tiers()
    print(f"Updated tiers for {len(tier_data)} subnets")
