#!/bin/bash
# Cron script to update subnet sentiment dashboard data
# Recommended to run every 12 hours via crontab:
# 0 */12 * * * /home/richie/discord/update_dashboard.sh >> /home/richie/discord/cron.log 2>&1

# Change to script directory
cd "$(dirname "$0")"

# Load environment variables if .env exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Log start time
echo "[$(date)] Starting dashboard update..."

# Run the analysis on all subnets
python3 run_all_subnet_analysis.py

# Check if the dashboard is built and accessible
if [ -d "subnet-sentiment-dashboard" ]; then
  echo "[$(date)] Dashboard data updated successfully!"
else
  echo "[$(date)] Warning: Dashboard directory not found!"
fi

# Log completion
echo "[$(date)] Update completed"
