from flask import Flask, jsonify, render_template, request
import pandas as pd
import feedparser
# Remove transformers import for production
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import re
from collections import defaultdict, Counter
import time
import threading
import queue
import os

app = Flask(__name__)

# Fallback data (no CSV dependency)
def get_fallback_data():
    return pd.DataFrame({
        'COMPANY_NAME': ['Reliance Industries', 'TCS', 'HDFC Bank', 'Infosys', 'ICICI Bank'],
        'SYMBOL': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK'],
        'SECTOR': ['Oil & Gas', 'IT', 'Banking', 'IT', 'Banking'],
        'LTP': [2500.0, 3400.0, 1650.0, 1800.0, 900.0],
        'CHNG': [50.0, 45.0, 25.0, -30.0, -15.0],
        '%CHNG': [2.0, 1.3, 1.5, -1.6, -1.6]
    })

# Load company data with fallback
try:
    if os.path.exists('nifty500.csv'):
        company_df = pd.read_csv('nifty500.csv')
        nifty_data = pd.read_csv('nifty500.csv')
    else:
        company_df = get_fallback_data()
        nifty_data = get_fallback_data()
    print(f"✅ Loaded {len(company_df)} companies")
except Exception as e:
    print(f"⚠️ Using fallback data: {e}")
    company_df = get_fallback_data()
    nifty_data = get_fallback_data()

# Remove AI models for production
sentiment_pipeline = None
summarizer = None

# Simplified RSS feeds (fewer sources for reliability)
RSS_FEEDS = {
    "economic_times_market": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "economic_times_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
    "financial_express": "https://www.financialexpress.com/market/feed/",
    "livemint": "https://www.livemint.com/rss/markets",
    "ndtv_business": "https://feeds.feedburner.com/ndtvprofit-latest",
    "zeebiz": "https://www.zeebiz.com/rss/markets.xml",
    "google_india_business": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp0Y0RvU0FtVnVHZ0pKVGtnQVAB?hl=en-IN&gl=IN&ceid=IN:en",
    "google_india_stocks": "https://news.google.com/rss/search?q=indian%20stocks&hl=en-IN&gl=IN&ceid=IN:en"
}

# Sector keywords (same as yours)
SECTOR_KEYWORDS = {
    "Banking": {
        "companies": ['hdfc bank', 'icici bank', 'sbi', 'state bank', 'axis bank', 'kotak mahindra',
                     'yes bank', 'indusind bank', 'federal bank', 'rbl bank', 'idfc first',
                     'hdfc', 'icici', 'kotak', 'axis', 'canara bank', 'pnb', 'punjab national'],
        "keywords": ['banking', 'bank', 'finance', 'loans', 'deposits', 'npa', 'credit', 
                    'financial services', 'nbfc', 'interest rates', 'rbi policy', 'repo rate'],
        "symbols": ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK']
    },
    
    "IT": {
        "companies": ['tcs', 'tata consultancy', 'infosys', 'wipro', 'hcl tech', 'hcltech',
                     'tech mahindra', 'mindtree', 'mphasis', 'ltts', 'cognizant', 'accenture'],
        "keywords": ['software', 'technology', 'information technology', 'digital', 
                    'tech services', 'it services', 'programming', 'cloud computing', 'ai'],
        "symbols": ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM']
    },
    
    "Oil & Gas": {
        "companies": ['reliance industries', 'reliance', 'ongc', 'oil and natural gas corporation', 
                     'bpcl', 'bharat petroleum', 'ioc', 'indian oil corporation', 'indian oil',
                     'gail', 'oil india', 'mrpl', 'hpcl', 'hindustan petroleum', 'gspl',
                     'igl', 'indraprastha gas', 'petronet lng', 'mangalore refinery'],
        "keywords": ['oil', 'gas', 'petroleum', 'refinery', 'crude oil', 'energy', 'petrol',
                    'diesel', 'lng', 'cng', 'natural gas', 'petrochemical', 'fuel', 'opec',
                    'drilling', 'exploration'],
        "symbols": ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL', 'OIL', 'MRPL', 'HPCL']
    }
}

# Global logs
processing_logs = []
log_lock = threading.Lock()

def add_log(message):
    with log_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        processing_logs.append(log_message)
        print(log_message)
        if len(processing_logs) > 30:  # Reduced log size
            processing_logs.pop(0)

def get_logs():
    with log_lock:
        return processing_logs.copy()

# Simplified sentiment analysis (keyword-based only)
def simple_sentiment_analysis(text, title=""):
    positive_words = ['profit', 'growth', 'up', 'rise', 'gain', 'bullish', 'positive']
    negative_words = ['loss', 'down', 'fall', 'decline', 'bearish', 'negative']
    
    combined_text = f"{title} {text}".lower()
    
    pos_count = sum(1 for word in positive_words if word in combined_text)
    neg_count = sum(1 for word in negative_words if word in combined_text)
    
    if pos_count > neg_count:
        return "Positive", 0.7
    elif neg_count > pos_count:
        return "Negative", 0.7
    else:
        return "Neutral", 0.5

def classify_sector(title, description):
    text = f"{title} {description}".lower()
    
    for sector, data in SECTOR_KEYWORDS.items():
        score = 0
        for company in data['companies']:
            if company in text:
                score += 5
        for keyword in data['keywords']:
            if keyword in text:
                score += 2
        
        if score >= 3:
            return sector
    
    # Check for general market indicators
    if any(word in text for word in ['sensex', 'nifty', 'market', 'stock']):
        return "Indian Markets"
    
    return None

def process_feed(feed_name, feed_url, max_articles=10):
    """Simplified feed processing"""
    try:
        add_log(f"🔄 Processing {feed_name}...")
        
        feed = feedparser.parse(feed_url)
        if not feed.entries:
            return {}
        
        sector_articles = defaultdict(list)
        processed = 0
        
        for entry in feed.entries[:max_articles]:
            title = entry.get('title', '')
            link = entry.get('link', '')
            description = BeautifulSoup(entry.get('summary', ''), 'html.parser').get_text()
            
            if not title or not link:
                continue
            
            sector = classify_sector(title, description)
            if sector:
                sentiment_label, sentiment_score = simple_sentiment_analysis(description, title)
                
                article_data = {
                    'title': title,
                    'description': description,
                    'url': link,
                    'sentiment': sentiment_score,
                    'sentiment_label': sentiment_label,
                    'source': feed_name.replace('_', ' ').title(),
                    'stock_mentions': []
                }
                
                sector_articles[sector].append(article_data)
                processed += 1
        
        add_log(f"✅ {feed_name}: {processed} articles")
        return dict(sector_articles)
        
    except Exception as e:
        add_log(f"❌ Error in {feed_name}: {e}")
        return {}

def fetch_news():
    """Simplified news fetching"""
    add_log("🚀 Starting news collection...")
    
    all_articles = defaultdict(list)
    
    for feed_name, feed_url in RSS_FEEDS.items():
        articles = process_feed(feed_name, feed_url)
        for sector, sector_articles in articles.items():
            all_articles[sector].extend(sector_articles)
        time.sleep(1)  # Rate limiting
    
    add_log("🎉 News collection completed!")
    return dict(all_articles)

def get_top_gainers():
    """Get top gainers with fallback"""
    try:
        data = nifty_data.copy()
        data = data.dropna(subset=["LTP", "%CHNG"])
        data["%CHNG"] = pd.to_numeric(data["%CHNG"], errors="coerce")
        gainers = data.sort_values(by="%CHNG", ascending=False).head(5)
        
        return [{
            "name": row.get("SYMBOL", "Unknown"),
            "price": f"₹{row['LTP']:.2f}",
            "change": f"₹{row.get('CHNG', 0):.2f}",
            "percent": f"{row['%CHNG']:.2f}%",
            "sector": row.get("SECTOR", "Unknown")
        } for _, row in gainers.iterrows()]
    except Exception as e:
        add_log(f"⚠️ Using sample gainer data: {e}")
        return [
            {"name": "RELIANCE", "price": "₹2500", "change": "₹50", "percent": "2.0%", "sector": "Oil & Gas"}
        ]

def get_top_losers():
    """Get top losers with fallback"""
    try:
        data = nifty_data.copy()
        data = data.dropna(subset=["LTP", "%CHNG"])
        data["%CHNG"] = pd.to_numeric(data["%CHNG"], errors="coerce")
        losers = data.sort_values(by="%CHNG", ascending=True).head(5)
        
        return [{
            "name": row.get("SYMBOL", "Unknown"),
            "price": f"₹{row['LTP']:.2f}",
            "change": f"₹{row.get('CHNG', 0):.2f}",
            "percent": f"{row['%CHNG']:.2f}%",
            "sector": row.get("SECTOR", "Unknown")
        } for _, row in losers.iterrows()]
    except Exception as e:
        add_log(f"⚠️ Using sample loser data: {e}")
        return [
            {"name": "TCS", "price": "₹3200", "change": "₹-45", "percent": "-1.4%", "sector": "IT"}
        ]

def generate_insights(sector_articles):
    """Generate sector insights"""
    insights = {}
    
    for sector, articles in sector_articles.items():
        if not articles:
            continue
        
        positive = [a for a in articles if a['sentiment_label'] == 'Positive']
        negative = [a for a in articles if a['sentiment_label'] == 'Negative']
        neutral = [a for a in articles if a['sentiment_label'] == 'Neutral']
        
        # Simple prediction
        if len(positive) > len(negative):
            prediction = "Likely Up 📈"
            confidence = f"{min(len(positive) * 20, 80)}%"
        elif len(negative) > len(positive):
            prediction = "Likely Down 📉"
            confidence = f"{min(len(negative) * 20, 80)}%"
        else:
            prediction = "Sideways 📊"
            confidence = "50%"
        
        insights[sector] = {
            "total_articles": len(articles),
            "positive_count": len(positive),
            "negative_count": len(negative),
            "neutral_count": len(neutral),
            "prediction": prediction,
            "prediction_confidence": confidence,
            "key_stocks": [],
            "trending_stocks": [],
            "latest_articles": articles[:5]
        }
    
    return insights

@app.route("/")
def dashboard():
    """Main dashboard route"""
    try:
        add_log("🏠 Loading dashboard...")
        
        sector_articles = fetch_news()
        sector_insights = generate_insights(sector_articles)
        total_articles = sum(len(articles) for articles in sector_articles.values())
        
        add_log(f"✅ Dashboard loaded with {total_articles} articles")
        
        return render_template(
            "complete_dashboard.html",
            gainers=get_top_gainers(),
            losers=get_top_losers(),
            sector_articles=sector_articles,
            sector_insights=sector_insights,
            total_articles=total_articles,
            logs=get_logs()[-10:],
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as e:
        add_log(f"❌ Dashboard error: {str(e)}")
        return f"<h1>Dashboard Loading...</h1><p>Please refresh in a moment.</p>"

@app.route("/api/logs")
def api_logs():
    """API endpoint for logs"""
    return jsonify({"logs": get_logs()})

@app.route("/summarize")
def summarize_url():
    """Simple summarization without AI"""
    url = request.args.get("url")
    if not url:
        return jsonify({"summary": "❌ No URL provided."})
    
    try:
        add_log(f"📄 Summarizing: {url[:50]}...")
        
        # Simple extraction
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; NewsBot/1.0)'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        text = soup.get_text()
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        
        # Simple extractive summary
        summary = '. '.join(sentences[:3]) + '.' if len(sentences) >= 3 else "Could not generate summary."
        
        sentiment_label, sentiment_score = simple_sentiment_analysis(summary)
        
        return jsonify({
            "summary": summary,
            "stock_mentions": [],
            "sentiment": sentiment_label,
            "sentiment_score": f"{sentiment_score:.2f}",
            "word_count": len(text.split()),
            "analysis_success": True,
            "extraction_method": "BeautifulSoup"
        })
        
    except Exception as e:
        add_log(f"❌ Summarization error: {e}")
        return jsonify({
            "summary": f"Error: Could not process article - {str(e)}",
            "analysis_success": False
        })

# Production-ready entry point
if __name__ == "__main__":
    add_log("🚀 Starting Production-Ready Stock Market Dashboard")
    add_log("📊 Simplified for reliability")
    
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
