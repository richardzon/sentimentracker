#!/usr/bin/env python3
"""
Test script to verify Mistral sentiment analysis is working
"""

from enhanced_scraper import EnhancedDiscordScraper
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_mistral_sentiment():
    """Test Mistral sentiment analysis with sample messages"""
    
    scraper = EnhancedDiscordScraper()
    
    # Test messages with varying sentiment
    test_messages = [
        "I'm excited about the new subnet updates, this is amazing work!",
        "How do I configure the consensus parameters for my subnet?",
        "This is incredibly frustrating, nothing is working correctly.",
        "I've been testing my subnet for days and it's finally running perfectly.",
        "What's the optimal gas limit for the Avalanche C-Chain?",
        "Just crashed again, I'm about to give up on this project."
    ]
    
    print("Testing Mistral sentiment analysis via analyze_sentiment_with_ai...")
    sentiment_scores = await scraper.analyze_sentiment_with_ai(test_messages)
    
    # Display results in a table
    print("\n--- MISTRAL SENTIMENT ANALYSIS RESULTS ---")
    print("| {:<60} | {:<10} |".format("Message", "Score"))
    print("|" + "-" * 62 + "|" + "-" * 12 + "|")
    
    for i, msg in enumerate(test_messages):
        # Truncate message if too long
        display_msg = msg[:57] + "..." if len(msg) > 57 else msg
        print("| {:<60} | {:<10.2f} |".format(
            display_msg, 
            sentiment_scores[i] if i < len(sentiment_scores) else 0
        ))

# Run the test
if __name__ == "__main__":
    asyncio.run(test_mistral_sentiment())
