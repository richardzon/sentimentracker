#!/usr/bin/env python3
"""
Script to monitor sentiment analysis on real subnet data
"""
import os
import sys
import json
import time
import asyncio
import logging
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from enhanced_scraper import EnhancedDiscordScraper
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_monitor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('SentimentMonitor')

# Make sure environment variables are loaded
load_dotenv()

class SentimentMonitor:
    def __init__(self):
        self.scraper = EnhancedDiscordScraper()
        self.results_dir = "sentiment_monitor_results"
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load subnet channels
        self.subnet_channels = self.load_subnet_channels()
        
        # Performance metrics
        self.performance_data = {
            "cached_time": [],
            "uncached_time": [],
            "messages_processed": [],
            "subnets_processed": []
        }
        
        # Sentiment distribution metrics
        self.sentiment_distribution = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
        
        # Technical message tracking
        self.technical_patterns = [
            "error", "bug", "fix", "issue", "problem", "crash", "exception",
            "traceback", "code", "function", "variable", "class", "method",
            "import", "module", "package", "library", "dependency", "install",
            "compile", "build", "deploy", "config", "configuration", "setting",
            "parameter", "argument", "debug", "test", "log", "```"
        ]
        self.technical_message_stats = {
            "total": 0,
            "classified_neutral": 0,
            "classified_positive": 0,
            "classified_negative": 0
        }
        
    def load_subnet_channels(self):
        """Load subnet channel mapping from file"""
        try:
            with open("subnet_channels.json", "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading subnet channels: {e}")
            return {}
    
    def is_technical_message(self, message):
        """Check if a message appears to be technical based on patterns"""
        # Extract message content if in format "hour:author:content"
        content = message.split(':', 2)[2] if len(message.split(':', 2)) >= 3 else message
        
        # Check for code blocks
        if "```" in content:
            return True
            
        # Check for technical patterns
        for pattern in self.technical_patterns:
            if pattern.lower() in content.lower():
                return True
                
        # Check for typical question patterns about technical topics
        question_starters = ["how do i", "how to", "can anyone", "is there a way", 
                             "what's the best", "what is the", "has anyone"]
        for starter in question_starters:
            if starter in content.lower():
                return True
                
        return False
        
    def categorize_sentiment(self, score):
        """Categorize sentiment score"""
        if score > 0.2:
            return "positive"
        elif score < -0.2:
            return "negative"
        else:
            return "neutral"
            
    async def analyze_subnet(self, subnet_num, clear_cache=False):
        """Analyze sentiment for a subnet and track metrics"""
        try:
            # Clear sentiment cache if requested (for comparison)
            if clear_cache:
                self.scraper.sentiment_cache = {}
                self.scraper.sentiment_cache_timestamp = {}
                
            # Get channel ID for the subnet
            subnet_data = self.subnet_channels.get(str(subnet_num))
            if not subnet_data:
                logger.warning(f"No channel ID found for subnet {subnet_num}")
                return None
                
            channel_id = subnet_data.get("channel_id")
            if not channel_id:
                logger.warning(f"Invalid channel data for subnet {subnet_num}")
                return None
                
            # Fetch messages
            messages = await self.scraper.scrape_channel_with_caching(subnet_num, channel_id, limit=100)
            if not messages:
                logger.warning(f"No messages found for subnet {subnet_num}")
                return None
                
            # Track start time
            start_time = time.time()
            
            # Analyze sentiment for each message
            sentiment_scores = await self.scraper.analyze_sentiment_with_ai(messages)
            
            # Track end time
            end_time = time.time()
            
            # Compute performance metrics
            processing_time = end_time - start_time
            
            # Record metrics for individual messages
            results = []
            for i, (message, score) in enumerate(zip(messages, sentiment_scores)):
                is_technical = self.is_technical_message(message)
                category = self.categorize_sentiment(score)
                
                # Track technical message stats
                if is_technical:
                    self.technical_message_stats["total"] += 1
                    if category == "neutral":
                        self.technical_message_stats["classified_neutral"] += 1
                    elif category == "positive":
                        self.technical_message_stats["classified_positive"] += 1
                    elif category == "negative":
                        self.technical_message_stats["classified_negative"] += 1
                
                # Track sentiment distribution for this subnet
                self.sentiment_distribution[f"subnet_{subnet_num}"][category] += 1
                
                results.append({
                    "message": message,
                    "sentiment_score": score,
                    "sentiment_category": category,
                    "is_technical": is_technical
                })
            
            # Return results along with metadata
            return {
                "subnet_num": subnet_num,
                "messages_count": len(messages),
                "processing_time": processing_time,
                "average_sentiment": sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
                "cached": not clear_cache,
                "results": results
            }
                
        except Exception as e:
            logger.error(f"Error analyzing subnet {subnet_num}: {e}")
            return None
    
    async def monitor_subnets(self, subnet_nums, include_uncached=True):
        """Monitor sentiment for multiple subnets and track performance"""
        all_results = []
        
        for subnet_num in subnet_nums:
            logger.info(f"Processing subnet {subnet_num}")
            
            # First run - should be uncached
            if include_uncached:
                logger.info(f"Analyzing subnet {subnet_num} without cache")
                uncached_results = await self.analyze_subnet(subnet_num, clear_cache=True)
                if uncached_results:
                    uncached_time = uncached_results["processing_time"]
                    self.performance_data["uncached_time"].append(uncached_time)
                    self.performance_data["messages_processed"].append(uncached_results["messages_count"])
                    logger.info(f"Subnet {subnet_num} (uncached): {uncached_results['messages_count']} messages in {uncached_time:.2f}s")
                    all_results.append(uncached_results)
            
            # Second run - should use cache
            logger.info(f"Analyzing subnet {subnet_num} with cache")
            cached_results = await self.analyze_subnet(subnet_num, clear_cache=False)
            if cached_results:
                cached_time = cached_results["processing_time"]
                self.performance_data["cached_time"].append(cached_time)
                logger.info(f"Subnet {subnet_num} (cached): {cached_results['messages_count']} messages in {cached_time:.2f}s")
                all_results.append(cached_results)
                
                # Track subnets processed
                self.performance_data["subnets_processed"].append(subnet_num)
        
        return all_results
    
    def generate_performance_report(self):
        """Generate performance report based on collected metrics"""
        if not self.performance_data["cached_time"] or not self.performance_data["uncached_time"]:
            return "Insufficient data for performance report"
        
        avg_uncached_time = sum(self.performance_data["uncached_time"]) / len(self.performance_data["uncached_time"])
        avg_cached_time = sum(self.performance_data["cached_time"]) / len(self.performance_data["cached_time"])
        total_messages = sum(self.performance_data["messages_processed"])
        speedup = avg_uncached_time / avg_cached_time if avg_cached_time > 0 else float('inf')
        
        report = [
            "=" * 60,
            "SENTIMENT ANALYSIS PERFORMANCE REPORT",
            "=" * 60,
            f"Subnets processed: {len(self.performance_data['subnets_processed'])}",
            f"Total messages: {total_messages}",
            f"Average uncached processing time: {avg_uncached_time:.2f}s",
            f"Average cached processing time: {avg_cached_time:.2f}s",
            f"Speedup from caching: {speedup:.2f}x",
            "",
            "TECHNICAL MESSAGE CLASSIFICATION",
            "-" * 60,
            f"Total technical messages: {self.technical_message_stats['total']}",
            f"  - Classified as neutral: {self.technical_message_stats['classified_neutral']} ({self.technical_message_stats['classified_neutral'] / self.technical_message_stats['total'] * 100:.2f}% if self.technical_message_stats['total'] > 0 else 'N/A')",
            f"  - Classified as positive: {self.technical_message_stats['classified_positive']} ({self.technical_message_stats['classified_positive'] / self.technical_message_stats['total'] * 100:.2f}% if self.technical_message_stats['total'] > 0 else 'N/A')",
            f"  - Classified as negative: {self.technical_message_stats['classified_negative']} ({self.technical_message_stats['classified_negative'] / self.technical_message_stats['total'] * 100:.2f}% if self.technical_message_stats['total'] > 0 else 'N/A')",
            "=" * 60
        ]
        
        return "\n".join(report)
    
    def plot_performance_metrics(self):
        """Plot performance metrics as charts"""
        # Create a multi-part figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Processing time comparison (cached vs uncached)
        if self.performance_data["cached_time"] and self.performance_data["uncached_time"]:
            axs[0, 0].bar(['Uncached', 'Cached'], 
                          [np.mean(self.performance_data["uncached_time"]), 
                           np.mean(self.performance_data["cached_time"])])
            axs[0, 0].set_title('Average Processing Time')
            axs[0, 0].set_ylabel('Time (seconds)')
        
        # 2. Speedup per subnet
        if self.performance_data["cached_time"] and self.performance_data["uncached_time"] and len(self.performance_data["cached_time"]) == len(self.performance_data["uncached_time"]):
            speedups = [u/c if c > 0 else 0 for u, c in zip(self.performance_data["uncached_time"], self.performance_data["cached_time"])]
            subnets = self.performance_data["subnets_processed"]
            if subnets and speedups:
                axs[0, 1].bar([f'Subnet {s}' for s in subnets], speedups)
                axs[0, 1].set_title('Speedup by Subnet')
                axs[0, 1].set_ylabel('Speedup Factor (x)')
        
        # 3. Technical message classification 
        if self.technical_message_stats["total"] > 0:
            techs = ['Neutral', 'Positive', 'Negative']
            counts = [self.technical_message_stats["classified_neutral"], 
                     self.technical_message_stats["classified_positive"],
                     self.technical_message_stats["classified_negative"]]
            axs[1, 0].pie(counts, labels=techs, autopct='%1.1f%%')
            axs[1, 0].set_title('Technical Message Classification')
        
        # 4. Overall sentiment distribution
        if self.sentiment_distribution:
            # Aggregate sentiment across all subnets
            overall_sentiment = {"positive": 0, "neutral": 0, "negative": 0}
            for subnet, counts in self.sentiment_distribution.items():
                overall_sentiment["positive"] += counts["positive"]
                overall_sentiment["neutral"] += counts["neutral"]
                overall_sentiment["negative"] += counts["negative"]
                
            labels = ['Positive', 'Neutral', 'Negative']
            sizes = [overall_sentiment["positive"], overall_sentiment["neutral"], overall_sentiment["negative"]]
            if sum(sizes) > 0:
                axs[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%')
                axs[1, 1].set_title('Overall Sentiment Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/performance_metrics.png")
        logger.info(f"Performance metrics chart saved to {self.results_dir}/performance_metrics.png")
        
    def save_results(self, results, filename=None):
        """Save analysis results to a file"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{timestamp}.json"
            
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {filepath}")
        return filepath

async def main():
    monitor = SentimentMonitor()
    
    # Select subnets to monitor - you can change these to focus on specific subnets
    subnet_nums = [1, 2, 3, 4, 5]  # Add more subnets as needed
    
    logger.info(f"Starting sentiment monitoring for subnets: {subnet_nums}")
    
    # Process subnets and track metrics
    results = await monitor.monitor_subnets(subnet_nums)
    
    # Generate performance report
    report = monitor.generate_performance_report()
    logger.info(f"\n{report}")
    
    # Plot metrics
    monitor.plot_performance_metrics()
    
    # Save results
    monitor.save_results(results)
    
    # Also save performance report
    with open(f"{monitor.results_dir}/performance_report.txt", 'w') as f:
        f.write(report)
    
    logger.info("Sentiment monitoring completed")

if __name__ == "__main__":
    # Run the monitor
    asyncio.run(main())
