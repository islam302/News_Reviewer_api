#!/bin/bash

echo "🚀 Starting Deployment for News_Reviewer_api..."

# Activate virtual environment
echo "📦 Activating virtual environment..."
source env/bin/activate

# Pull latest code
echo "⬇️ Pulling latest code from GitHub..."
git pull origin main

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Run migrations
echo "🗃️ Running migrations..."
python manage.py migrate

# Collect static files
echo "🎨 Collecting static files..."
python manage.py collectstatic --noinput

# Restart Gunicorn service
echo "🔄 Restarting gunicorn service..."
sudo systemctl restart news_reviewer

# Show service status
echo "📊 Checking service status..."
sudo systemctl status news_reviewer --no-pager

echo "✅ Deployment Finished!"
