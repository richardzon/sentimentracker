# Sentiment Tracker for Discord Subnets

A complete system for tracking, analyzing, and visualizing sentiment across Discord subnet communities.

## Overview

This project monitors sentiment across Discord subnet communities by analyzing messages and classifying their sentiment using AI. It includes:

- Scraping Discord messages with efficient caching
- Sentiment analysis using the Mistral AI model through OpenRouter API
- Technical question detection to ensure neutral classification of technical content
- Sentiment caching to avoid redundant API calls
- Data visualization dashboard for sentiment trends

## Key Features

### 1. Optimized Sentiment Analysis
- Uses Mistral AI model via OpenRouter for sentiment analysis
- Properly identifies technical questions as neutral
- Previously used TextBlob for preliminary analysis

### 2. Efficient Caching System
- Caches message content to reduce Discord API calls
- Stores sentiment scores to minimize expensive AI API calls
- Supports automatic and manual refresh cycles

### 3. Interactive Dashboard
- Visual representation of sentiment across subnets
- Displays sentiment distributions and trends
- Sortable by various metrics

## Installation

### Prerequisites
- Python 3.9+
- Node.js and npm (for dashboard)
- Discord User Token
- OpenRouter API Key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/richardzon/sentimentracker.git
cd sentimentracker
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in a `.env` file:
```
DISCORD_USER_TOKEN=your_discord_user_token
OPENROUTER_API_KEY=your_openrouter_api_key
```

4. Set up the dashboard:
```bash
cd subnet-sentiment-dashboard
npm install
npm start
```

## Usage

### Initial Setup & Data Collection

1. Extract subnet channels:
```bash
python extract_subnet_channels.py
```

2. Reset database and scrape messages:
```bash
python reset_and_scrape.py
```

3. Analyze sentiment:
```bash
python run_all_subnet_analysis.py
```

4. Update dashboard data:
```bash
./update_dashboard.sh
```

### Regular Monitoring

For ongoing monitoring, use:
```bash
python monitor_sentiment.py
```

### Refresh All Data

To refresh all data:
```bash
python reset_scrape_analyze.py
```

## Configuration Files

- `subnet_channels.json`: Maps subnet numbers to channel IDs
- `subnet_ids.json`: Contains subnet IDs and names
- `.env`: Environment variables for API keys

## Architecture

### Components

1. **Scraper (`enhanced_scraper.py`)**: Handles Discord message scraping and caching
2. **Sentiment Analyzer**: Processes message sentiment using Mistral AI
3. **Dashboard**: Visualizes sentiment data

### Data Flow

1. Messages are scraped from Discord and stored in SQLite
2. Sentiment analysis is performed on messages, with results cached
3. Dashboard displays sentiment data from JSON files

## Dashboard

The dashboard visualizes sentiment data and allows for sorting and filtering by:
- Subnet name
- Average sentiment
- Message volume
- Sentiment distribution

## License

MIT

## Acknowledgements

- OpenRouter for providing access to Mistral AI
- Discord for the communication platform
