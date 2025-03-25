#!/usr/bin/env python3
"""
Script to rebuild the all_subnets_latest.json file from individual subnet files
This will restore the dashboard functionality after it was accidentally cleared
"""

import os
import json
import glob
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DashboardRebuilder")

def rebuild_dashboard_data():
    """Rebuild the all_subnets_latest.json file from individual subnet files"""
    # Path to the dashboard data directory
    dashboard_dir = "subnet-sentiment-dashboard/public/data"
    
    # Path to the all_subnets_latest.json file
    all_subnets_file = os.path.join(dashboard_dir, "all_subnets_latest.json")
    
    # Initialize the all_subnets data structure
    all_subnets_data = {}
    
    # Find all individual subnet latest files
    subnet_files = glob.glob(os.path.join(dashboard_dir, "subnet_*_latest.json"))
    logger.info(f"Found {len(subnet_files)} individual subnet files")
    
    # Process each subnet file
    for subnet_file in subnet_files:
        try:
            # Extract subnet number from filename
            subnet_num = os.path.basename(subnet_file).split('_')[1]
            
            # Read the subnet data
            with open(subnet_file, 'r') as f:
                subnet_data = json.load(f)
            
            # Add to the all_subnets data
            all_subnets_data[subnet_num] = subnet_data
            logger.info(f"Added subnet {subnet_num} data to all_subnets")
            
        except Exception as e:
            logger.error(f"Error processing {subnet_file}: {e}")
    
    # Save the rebuilt all_subnets data
    try:
        # Create a backup of the existing file if it exists
        if os.path.exists(all_subnets_file):
            backup_file = all_subnets_file + ".bak"
            logger.info(f"Creating backup of existing file: {backup_file}")
            try:
                with open(all_subnets_file, 'r') as f:
                    existing_data = json.load(f)
                with open(backup_file, 'w') as f:
                    json.dump(existing_data, f, indent=2)
            except:
                logger.warning(f"Could not create backup, continuing anyway")
        
        # Write the new data using atomic pattern
        temp_file = all_subnets_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(all_subnets_data, f, indent=2)
        
        # Replace the original file
        os.replace(temp_file, all_subnets_file)
        logger.info(f"Successfully rebuilt {all_subnets_file} with {len(all_subnets_data)} subnets")
        
    except Exception as e:
        logger.error(f"Error saving rebuilt data to {all_subnets_file}: {e}")
        return False
    
    return True

if __name__ == "__main__":
    logger.info("Starting dashboard data rebuild...")
    success = rebuild_dashboard_data()
    
    if success:
        logger.info("Dashboard data rebuild completed successfully")
        print("Dashboard data has been restored! The sentiment scores should now be visible again.")
    else:
        logger.error("Dashboard data rebuild failed")
        print("Error rebuilding dashboard data. Check the logs for details.")
