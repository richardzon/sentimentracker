#!/usr/bin/env python3
"""
extract_subnet_channels.py

This script extracts channel IDs for all subnets from the Bittensor Discord server,
specifically from the provided subnet category IDs.
"""

import os
import json
import aiohttp
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Discord API constants
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
API_BASE_URL = 'https://discord.com/api/v10'
HEADERS = {
    'Authorization': f'Bot {DISCORD_TOKEN}',
    'Content-Type': 'application/json',
    'User-Agent': 'SubnetSentimentBot/1.0'
}

# Fixed Bittensor guild parameters
GUILD_NAME = "Bittensor"
SUBNET_CATEGORY_IDS = [
    "1161764488186441768",  # Subnets category
    "1290321693427892358"   # Subnets2 category
]

async def get_guild_channels(guild_id):
    """Fetch all channels in a specific guild."""
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        async with session.get(f'{API_BASE_URL}/guilds/{guild_id}/channels') as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Error fetching channels for guild {guild_id}: {response.status}")
                print(await response.text())
                return []

async def find_guild_id(guild_name):
    """Find the guild ID based on the guild name."""
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        async with session.get(f'{API_BASE_URL}/users/@me/guilds') as response:
            if response.status == 200:
                guilds = await response.json()
                for guild in guilds:
                    if guild['name'] == guild_name:
                        return guild['id']
                print(f"Guild '{guild_name}' not found in the list of accessible guilds.")
                return None
            else:
                print(f"Error fetching guilds: {response.status}")
                print(await response.text())
                return None

def get_subnet_number(channel_name):
    """Extract the subnet number from the channel name."""
    name = channel_name.lower()
    
    # Pattern: subnet-XX or s-XX
    if name.startswith('subnet-'):
        parts = name[7:].split('-')
        if parts[0].isdigit():
            return int(parts[0])
    elif name.startswith('s'):
        parts = name[1:].split('-')
        if parts[0].isdigit():
            return int(parts[0])
    
    # Try to find a number in the channel name
    for part in name.split('-'):
        if part.isdigit():
            return int(part)
    
    # If no clear pattern, return None
    return None

async def main():
    print(f"Looking for {GUILD_NAME} guild...")
    
    # Find the Bittensor guild ID
    guild_id = await find_guild_id(GUILD_NAME)
    
    if not guild_id:
        print("Could not find the guild ID. Please check your bot token and permissions.")
        return
    
    print(f"Found {GUILD_NAME} guild with ID: {guild_id}")
    
    # Fetch all channels in the guild
    channels = await get_guild_channels(guild_id)
    
    if not channels:
        print("No channels found or API error occurred.")
        return
    
    # Find channels in the specified subnet categories
    subnet_channels = []
    
    for channel in channels:
        # Only include text channels that are under the specified categories
        if channel['type'] == 0 and channel.get('parent_id') in SUBNET_CATEGORY_IDS:
            subnet_channels.append(channel)
            print(f"Found subnet channel: {channel['name']} (ID: {channel['id']})")
    
    if not subnet_channels:
        print("No subnet channels found in the specified categories.")
        print(f"Make sure the category IDs {SUBNET_CATEGORY_IDS} are correct.")
        return
    
    # Sort and organize subnet channels
    organized_subnets = {}
    
    for channel in subnet_channels:
        subnet_num = get_subnet_number(channel['name'])
        if subnet_num is not None:
            organized_subnets[subnet_num] = {
                'channel_id': channel['id'],
                'name': channel['name'],
                'position': channel.get('position', 0)
            }
            print(f"Identified Subnet {subnet_num}: {channel['name']} (ID: {channel['id']})")
    
    # Sort by subnet number
    sorted_subnets = {k: organized_subnets[k] for k in sorted(organized_subnets.keys())}
    
    # Display and save results
    print(f"\nSuccessfully extracted {len(sorted_subnets)} subnet channels:")
    for subnet_num, data in sorted_subnets.items():
        print(f"Subnet {subnet_num}: {data['name']} (ID: {data['channel_id']})")
    
    # Save to file
    with open('subnet_channels.json', 'w') as f:
        json.dump(sorted_subnets, f, indent=2)
    
    print(f"\nSaved subnet channel data to subnet_channels.json")
    
    # Also save in format suitable for the dashboard
    subnet_ids = {str(subnet_num): data['channel_id'] for subnet_num, data in sorted_subnets.items()}
    with open('subnet_ids.json', 'w') as f:
        json.dump(subnet_ids, f, indent=2)
    
    print(f"Saved simplified subnet IDs to subnet_ids.json")

if __name__ == '__main__':
    asyncio.run(main())
