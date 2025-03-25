#!/usr/bin/env python3
"""
show_subnet_ratings.py

This script shows the current time-based ratings for subnets based on when "const" 
last made positive comments.

Rating system:
- Emerald: Positive comment within last hour
- Golden: Positive comment within last 6 hours
- Silver: Positive comment within last 24 hours
- Bronze: Positive comment within last 30 hours
"""

import sqlite3
import json
from datetime import datetime, timedelta
import os
import sys

DATABASE_PATH = 'discord_messages.db'

def calculate_rating(timestamp_str):
    """Calculate time-based rating from timestamp."""
    try:
        # Parse the timestamp
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.now(timestamp.tzinfo)
        
        # Calculate hours difference
        hours_ago = (now - timestamp).total_seconds() / 3600
        
        # Determine rating based on recency
        if hours_ago <= 1:
            return 'emerald', hours_ago
        elif hours_ago <= 6:
            return 'golden', hours_ago
        elif hours_ago <= 24:
            return 'silver', hours_ago  
        elif hours_ago <= 30:
            return 'bronze', hours_ago
        else:
            return None, hours_ago
            
    except Exception as e:
        print(f"Error calculating rating: {e}")
        return None, 0

def main():
    """Show current subnet ratings."""
    print("=" * 80)
    print(f"SUBNET RATINGS DASHBOARD - Current time: {datetime.now().isoformat()}")
    print("=" * 80)
    print("\nRating system:")
    print("  🟢 EMERALD: Const spoke positively within the last hour")
    print("  🟡 GOLDEN: Const spoke positively within the last 6 hours")
    print("  ⚪ SILVER: Const spoke positively within the last 24 hours")
    print("  🟠 BRONZE: Const spoke positively within the last 30 hours")
    print("  ⚫ NONE: No positive messages from const in the last 30 hours")
    
    # Connect to the database
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all subnets where "const" has spoken positively
    cursor.execute("""
        SELECT subnet_num, MAX(timestamp) as latest_positive
        FROM messages 
        WHERE author LIKE 'const%' 
            AND sentiment_score > 0.3
        GROUP BY subnet_num
        ORDER BY latest_positive DESC
    """)
    
    results = cursor.fetchall()
    
    if not results:
        print("\nNo subnets found with positive messages from const.")
        return
    
    # Show ratings
    print(f"\nFound {len(results)} subnets with positive messages from const:\n")
    
    for row in results:
        subnet_num = row['subnet_num']
        latest_timestamp = row['latest_positive']
        
        rating, hours_ago = calculate_rating(latest_timestamp)
        rating_icon = "🟢" if rating == "emerald" else "🟡" if rating == "golden" else "⚪" if rating == "silver" else "🟠" if rating == "bronze" else "⚫"
        rating_text = rating.upper() if rating else "NONE"
        
        print(f"{rating_icon} Subnet {subnet_num}: {rating_text} ({hours_ago:.1f} hours ago)")
        
        # Get the positive message content
        cursor.execute("""
            SELECT content, sentiment_score
            FROM messages
            WHERE subnet_num = ? AND author LIKE 'const%' AND sentiment_score > 0.3
            ORDER BY timestamp DESC
            LIMIT 1
        """, (subnet_num,))
        
        message = cursor.fetchone()
        if message:
            content = message['content']
            score = message['sentiment_score']
            print(f"   Message: \"{content[:100]}{'...' if len(content) > 100 else ''}\"")
            print(f"   Sentiment score: {score:.2f}")
            
        print()
    
    print("\nNote: These ratings will be reflected in the dashboard automatically.")
    print("The dashboard refreshes every 2 hours to update these ratings.")
    
    conn.close()

if __name__ == "__main__":
    main()
