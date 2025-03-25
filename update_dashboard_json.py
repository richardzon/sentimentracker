#!/usr/bin/env python3
"""
Quick script to rebuild the all_subnets_latest.json file from the latest subnet data
"""
import os
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dashboard_updater")

# Constants
DASHBOARD_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "subnet-sentiment-dashboard", "public", "data")
ALL_SUBNETS_FILE = os.path.join(DASHBOARD_DATA_DIR, "all_subnets_latest.json")
SUBNET_SENTIMENT_FILE = os.path.join(DASHBOARD_DATA_DIR, "subnet_sentiment.json")

def update_dashboard():
    """Force update the dashboard JSON files with the latest subnet data"""
    logger.info("Starting dashboard data update")
    
    # Get all available subnet data files
    all_subnet_data = {}
    subnet_sentiment_data = []
    current_time = datetime.now().isoformat()
    
    # Find all subnet JSON files (looking for both regular and _latest files)
    subnet_files = [f for f in os.listdir(DASHBOARD_DATA_DIR) 
                   if f.startswith("subnet_") and f.endswith(".json") 
                   and not f.endswith("_analysis.json")]
    
    logger.info(f"Found {len(subnet_files)} subnet data files")
    
    # Process each subnet file
    for file_name in subnet_files:
        try:
            file_path = os.path.join(DASHBOARD_DATA_DIR, file_name)
            
            # Skip old or history files
            if "history" in file_path:
                continue
                
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # For simple files, get just the basic data
            if "_latest" not in file_name:
                subnet_num = str(data.get("subnet"))
                
                # See if there's a corresponding _latest file with full data
                latest_file = os.path.join(DASHBOARD_DATA_DIR, f"subnet_{subnet_num}_latest.json")
                if os.path.exists(latest_file):
                    with open(latest_file, 'r') as f:
                        full_data = json.load(f)
                        
                    if subnet_num not in all_subnet_data:
                        all_subnet_data[subnet_num] = full_data
                        
                        # Also add to sentiment data array
                        subnet_sentiment_data.append({
                            "subnet": int(subnet_num),
                            "sentiment": full_data.get("average_sentiment", data.get("avg_sentiment", 0)),
                            "message_count": full_data.get("message_count", data.get("message_count", 0)),
                            "timestamp": current_time
                        })
                else:
                    # Use simple file data if no _latest exists
                    if subnet_num not in all_subnet_data:
                        # Create a simple entry with available data
                        all_subnet_data[subnet_num] = {
                            "subnet": int(subnet_num),
                            "average_sentiment": data.get("avg_sentiment", 0),
                            "message_count": data.get("message_count", 0),
                            "last_updated": data.get("updated_at", current_time),
                            "sentiment_distribution": {
                                "positive": 33,
                                "neutral": 34, 
                                "negative": 33
                            },
                            "technical_percentage": 60,
                            "key_topics": ["General discussion"],
                            "messages": []
                        }
                        
                        # Also add to sentiment data array
                        subnet_sentiment_data.append({
                            "subnet": int(subnet_num),
                            "sentiment": data.get("avg_sentiment", 0),
                            "message_count": data.get("message_count", 0),
                            "timestamp": data.get("updated_at", current_time)
                        })
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
    
    # Save the updated files
    logger.info(f"Writing updated data to {ALL_SUBNETS_FILE} with {len(all_subnet_data)} subnets")
    with open(ALL_SUBNETS_FILE, 'w') as f:
        json.dump(all_subnet_data, f, indent=2)
    
    logger.info(f"Writing updated sentiment data to {SUBNET_SENTIMENT_FILE}")
    with open(SUBNET_SENTIMENT_FILE, 'w') as f:
        json.dump(subnet_sentiment_data, f, indent=2)
    
    logger.info("Dashboard data update complete!")

if __name__ == "__main__":
    update_dashboard()
