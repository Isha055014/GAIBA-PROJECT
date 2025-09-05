from __future__ import annotations

import os
import io
import time
import json
import re
import math
from typing import List, Dict, Optional, Tuple, Any
import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from pypdf import PdfReader
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# -------------- Utilities --------------

def safe_getenv(key: str) -> Optional[str]:
    """Get secrets from Streamlit secrets or environment variables."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)

# Load API keys
GROQ_API_KEY = safe_getenv("GROQ_API_KEY")
NEWSAPI_KEY = safe_getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = safe_getenv("ALPHA_VANTAGE_KEY")
SEC_API_KEY = safe_getenv("SEC_API_KEY")

# -------------- Document Processing --------------

class AdvancedTextSplitter:
    """Advanced text splitter that respects sentence boundaries"""
    def split_text(self, text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[str]:
        # Split by sentences first for better context preservation
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap
                overlap_start = max(0, len(current_chunk) - math.ceil(chunk_overlap / 20))
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) + 1 for s in current_chunk) - 1 if current_chunk else 0
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

class DocIngestor:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = AdvancedTextSplitter()

    def parse_file(self, file: io.BytesIO, filename: str) -> str:
        suffix = os.path.splitext(filename)[1].lower()
        try:
            if suffix == ".pdf":
                reader = PdfReader(file)
                text = "\n".join([p.extract_text() or "" for p in reader.pages])
                return text
            elif suffix in {".txt", ".md"}:
                return file.read().decode(errors="ignore")
            elif suffix == ".csv":
                df = pd.read_csv(file)
                return df.to_csv(index=False)
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(file)
                return df.to_csv(index=False)
            else:
                return ""
        except Exception as e:
            st.warning(f"Error parsing file {filename}: {e}")
            return ""

    def chunk(self, text: str) -> List[str]:
        try:
            chunks = self.splitter.split_text(text, self.chunk_size, self.chunk_overlap)
            return [c.strip() for c in chunks if c.strip()]
        except Exception as e:
            st.warning(f"Error chunking text: {e}")
            return [text.strip()] if text.strip() else []

    def extract_financial_data(self, text: str) -> Dict[str, Any]:
        """Extract financial data from document text"""
        financial_data = {}
        
        # Look for revenue patterns
        revenue_patterns = [
            r'revenue[\s:]*\$?[\s]*([\d,]+(?:\.\d+)?)[\s]*(?:million|billion|M|B)?',
            r'total[\s]+revenue[\s:]*\$?[\s]*([\d,]+(?:\.\d+)?)[\s]*(?:million|billion|M|B)?',
            r'sales[\s:]*\$?[\s]*([\d,]+(?:\.\d+)?)[\s]*(?:million|billion|M|B)?'
        ]
        
        for pattern in revenue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the largest value found
                revenue_values = [float(match.replace(',', '')) for match in matches]
                financial_data['revenue'] = max(revenue_values)
                # Check if mentioned in millions/billions
                if any(word in text.lower() for word in ['billion', 'b']):
                    financial_data['revenue'] *= 1000  # Convert to millions
                break
        
        # Look for profit patterns
        profit_patterns = [
            r'net[\s]+income[\s:]*\$?[\s]*([\d,]+(?:\.\d+)?)[\s]*(?:million|billion|M|B)?',
            r'profit[\s:]*\$?[\s]*([\d,]+(?:\.\d+)?)[\s]*(?:million|billion|M|B)?',
            r'net[\s]+profit[\s:]*\$?[\s]*([\d,]+(?:\.\d+)?)[\s]*(?:million|billion|M|B)?'
        ]
        
        for pattern in profit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                profit_values = [float(match.replace(',', '')) for match in matches]
                financial_data['net_income'] = max(profit_values)
                if any(word in text.lower() for word in ['billion', 'b']):
                    financial_data['net_income'] *= 1000
                break
        
        return financial_data

# -------------- Financial Analysis --------------

class FinancialSnapshot:
    def __init__(self, ticker: str, **kwargs):
        self.ticker = ticker
        for key, value in kwargs.items():
            setattr(self, key, value)

class AdvancedFinanceFetcher:
    def __init__(self):
        self.alpha_vantage_key = ALPHA_VANTAGE_KEY

    def fetch_yahoo_data(self, ticker: str) -> FinancialSnapshot:
        """Fetch data from Yahoo Finance"""
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            history = t.history(period="1y")
            
            # Get current price from history if not in info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price and not history.empty:
                current_price = history['Close'].iloc[-1]
            
            # Calculate volatility (standard deviation of daily returns)
            volatility = None
            if not history.empty and len(history) > 30:
                returns = history['Close'].pct_change().dropna()
                volatility = returns.std() * math.sqrt(252)  # Annualized
            
            return FinancialSnapshot(
                ticker=ticker,
                price=current_price,
                market_cap=info.get("marketCap"),
                pe=info.get("trailingPE"),
                ps=info.get("priceToSalesTrailing12Months"),
                debt_to_equity=info.get("debtToEquity"),
                gross_margin=info.get("grossMargins"),
                operating_margin=info.get("operatingMargins"),
                revenue_ttm=info.get("totalRevenue"),
                ebitda=info.get("ebitda"),
                profit_margin=info.get("profitMargins"),
                beta=info.get("beta"),
                volatility=volatility,
                currency=info.get("currency"),
                sector=info.get("sector"),
                industry=info.get("industry")
            )
        except Exception as e:
            st.warning(f"Error fetching Yahoo data for {ticker}: {e}")
            return FinancialSnapshot(ticker=ticker)

    def fetch_alpha_vantage_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch additional data from Alpha Vantage"""
        if not self.alpha_vantage_key:
            return {}
        
        try:
            # Get income statement
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "INCOME_STATEMENT",
                "symbol": ticker,
                "apikey": self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            financials = {}
            if "annualReports" in data and data["annualReports"]:
                latest = data["annualReports"][0]
                financials = {
                    "revenue": float(latest.get("totalRevenue", 0)),
                    "gross_profit": float(latest.get("grossProfit", 0)),
                    "net_income": float(latest.get("netIncome", 0)),
                    "operating_income": float(latest.get("operatingIncome", 0))
                }
            
            return financials
        except Exception as e:
            st.warning(f"Error fetching Alpha Vantage data for {ticker}: {e}")
            return {}

    def fetch(self, ticker: str) -> FinancialSnapshot:
        """Fetch comprehensive financial data"""
        yahoo_data = self.fetch_yahoo_data(ticker)
        alpha_data = self.fetch_alpha_vantage_data(ticker)
        
        # Merge data
        for key, value in alpha_data.items():
            if not getattr(yahoo_data, key, None) and value:
                setattr(yahoo_data, key, value)
        
        return yahoo_data

    def calculate_financial_score(self, financial_a: FinancialSnapshot, financial_b: FinancialSnapshot, 
                                transaction_type: str) -> float:
        """Calculate financial compatibility score (0-1)"""
        scores = []
        
        # 1. Market Cap Compatibility (more important for acquisitions)
        if financial_a.market_cap and financial_b.market_cap:
            if transaction_type == "Acquisition":
                # For acquisitions, acquirer should be significantly larger
                size_ratio = financial_a.market_cap / financial_b.market_cap
                if size_ratio >= 5:
                    scores.append(0.9)  # Ideal size difference
                elif size_ratio >= 2:
                    scores.append(0.7)  # Good size difference
                elif size_ratio >= 1:
                    scores.append(0.5)  # Similar size - might be challenging
                else:
                    scores.append(0.3)  # Target larger than acquirer - risky
            else:
                # For mergers, similar size is better
                size_ratio = max(financial_a.market_cap, financial_b.market_cap) / min(financial_a.market_cap, financial_b.market_cap)
                if size_ratio <= 2:
                    scores.append(0.8)  # Similar size - good for merger
                elif size_ratio <= 5:
                    scores.append(0.6)  # Moderate size difference
                else:
                    scores.append(0.4)  # Very different sizes - challenging
        
        # 2. P/E Ratio Comparison
        if financial_a.pe and financial_b.pe:
            pe_ratio = min(financial_a.pe, financial_b.pe) / max(financial_a.pe, financial_b.pe)
            scores.append(pe_ratio * 0.8)  # Weighted importance
        
        # 3. Profit Margin Comparison
        if financial_a.profit_margin and financial_b.profit_margin:
            margin_diff = abs(financial_a.profit_margin - financial_b.profit_margin)
            if margin_diff <= 0.05:  # 5% difference
                scores.append(0.8)
            elif margin_diff <= 0.1:  # 10% difference
                scores.append(0.6)
            elif margin_diff <= 0.2:  # 20% difference
                scores.append(0.4)
            else:
                scores.append(0.2)
        
        # 4. Revenue Growth Compatibility (placeholder - would use historical data)
        scores.append(0.6)  # Moderate assumption
        
        # 5. Debt-to-Equity Comparison
        if financial_a.debt_to_equity and financial_b.debt_to_equity:
            de_diff = abs(financial_a.debt_to_equity - financial_b.debt_to_equity)
            if de_diff <= 0.5:  # Similar debt levels
                scores.append(0.7)
            elif de_diff <= 1.0:
                scores.append(0.5)
            else:
                scores.append(0.3)
        
        # Calculate weighted average
        if scores:
            # Apply weights based on importance
            weights = [0.25, 0.20, 0.20, 0.15, 0.20]
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights[:len(scores)]))
            total_weight = sum(weights[:len(scores)])
            return weighted_sum / total_weight
        
        return 0.5  # Default if no data

# -------------- News & Sentiment Analysis --------------

class AdvancedNewsFetcher:
    def __init__(self):
        self.newsapi_key = NEWSAPI_KEY

    def fetch_news(self, query: str, transaction_type: str, limit: int = 15) -> List[Dict]:
        items = []
        
        if not self.newsapi_key:
            st.info("NewsAPI key not configured. Using sample news data.")
            return [
                {"title": f"Sample news about {query} {transaction_type.lower()}", 
                 "description": "This is sample news content.", 
                 "source": "Sample News", 
                 "url": "#", 
                 "publishedAt": "2024-01-01",
                 "content": f"Sample content about {query} considering a {transaction_type.lower()}."},
                {"title": f"Market analysis for {query}", 
                 "description": "Sample market analysis content.", 
                 "source": "Market News", 
                 "url": "#", 
                 "publishedAt": "2024-01-01",
                 "content": f"Analysts are discussing the potential for {transaction_type.lower()} activity in the sector."}
            ]
        
        try:
            # First search for company-specific news
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f'"{query}" AND ({transaction_type.lower()} OR acquisition OR merger OR "strategic transaction")',
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": min(limit, 100),
                "apiKey": self.newsapi_key,
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            # Check if we need to upgrade (426 error)
            if response.status_code == 426:
                st.warning("NewsAPI requires a paid plan for comprehensive news access. Using alternative sources.")
                raise Exception("NewsAPI upgrade required")
                
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                for article in data.get("articles", [])[:limit]:
                    items.append({
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "url": article.get("url", ""),
                        "publishedAt": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", ""),
                        "content": article.get("content", "") or article.get("description", "")
                    })
            
            # If not enough results, broaden the search
            if len(items) < 5:
                params["q"] = f'"{query}" AND (business OR corporate OR financial)'
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == "ok":
                    for article in data.get("articles", [])[:limit-len(items)]:
                        items.append({
                            "title": article.get("title", ""),
                            "description": article.get("description", ""),
                            "url": article.get("url", ""),
                            "publishedAt": article.get("publishedAt", ""),
                            "source": article.get("source", {}).get("name", ""),
                            "content": article.get("content", "") or article.get("description", "")
                        })
                        
        except Exception as e:
            st.warning(f"NewsAPI request failed: {e}")
            items = [
                {"title": f"News about {query}", 
                 "description": "Could not fetch real news. Please check API key.", 
                 "source": "System", 
                 "url": "#", 
                 "publishedAt": "2024-01-01",
                 "content": f"Content about {query} would appear here if NewsAPI was configured."}
            ]
        
        return items

    def analyze_news_sentiment(self, news_items: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment of news articles using Groq API if available, otherwise fallback to simple analysis"""
        if not news_items:
            return {"score": 0.5, "risk_factors": [], "positive_factors": []}
        
        # Use Groq API for advanced sentiment analysis if available
        if GROQ_API_KEY:
            try:
                # Prepare news content for analysis
                news_text = "\n\n".join([
                    f"Title: {item['title']}\nContent: {item.get('content', item.get('description', ''))}"
                    for item in news_items[:5]  # Limit to 5 articles for token efficiency
                ])
                
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                prompt = f"""
                Analyze these news articles for sentiment regarding mergers and acquisitions. 
                Return a JSON response with:
                - overall_sentiment: a score between 0 (very negative) and 1 (very positive)
                - risk_factors: list of potential risks mentioned
                - positive_factors: list of positive aspects mentioned
                
                News articles:
                {news_text}
                """
                
                payload = {
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": "You are a financial analyst specializing in M&A sentiment analysis. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000
                }

                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Parse the JSON response
                content = data["choices"][0]["message"]["content"]
                # Extract JSON from the response (in case there's additional text)
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                
            except Exception as e:
                st.warning(f"Groq sentiment analysis failed: {e}")
        
        # Fallback: simple keyword-based analysis
        return self.simple_news_analysis(news_items)
    
    def simple_news_analysis(self, news_items: List[Dict]) -> Dict[str, Any]:
        """Simple keyword-based news analysis"""
        positive_keywords = {
            "growth", "profit", "gain", "success", "expansion", "opportunity", 
            "synergy", "strategic", "benefit", "advantage", "positive", "strong",
            "approve", "accept", "agree", "merge", "acquire", "buy", "purchase"
        }
        
        negative_keywords = {
            "loss", "decline", "drop", "fall", "risk", "problem", "issue", "concern",
            "challenge", "difficulty", "oppose", "reject", "deny", "against", "lawsuit",
            "investigation", "regulate", "probe", "fraud", "scandal", "negative", "weak"
        }
        
        risk_factors = []
        positive_factors = []
        sentiment_scores = []
        
        for item in news_items:
            content = f"{item.get('title', '')} {item.get('content', '')} {item.get('description', '')}".lower()
            
            # Count positive and negative keywords
            pos_count = sum(1 for word in positive_keywords if word in content)
            neg_count = sum(1 for word in negative_keywords if word in content)
            
            # Calculate sentiment score for this article
            total = pos_count + neg_count
            if total > 0:
                sentiment_scores.append(pos_count / total)
            else:
                sentiment_scores.append(0.5)  # Neutral if no keywords found
            
            # Extract potential risk factors
            if any(risk in content for risk in ["risk", "problem", "issue", "concern", "challenge", "lawsuit", "investigation"]):
                # Try to extract the context around risk words
                risk_factors.append(f"Risk mentioned in: {item.get('title', 'Unknown title')}")
            
            # Extract potential positive factors
            if any(pos in content for pos in ["growth", "opportunity", "synergy", "benefit", "advantage"]):
                positive_factors.append(f"Positive aspect in: {item.get('title', 'Unknown title')}")
        
        # Calculate overall sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
        
        return {
            "score": avg_sentiment,
            "risk_factors": risk_factors[:5],  # Limit to top 5
            "positive_factors": positive_factors[:5]  # Limit to top 5
        }

# -------------- Legal & Regulatory Analysis --------------

class LegalAnalyzer:
    def __init__(self):
        self.sec_api_key = SEC_API_KEY

    def check_sec_filings(self, ticker: str) -> Dict[str, Any]:
        """Check SEC filings for legal issues"""
        if not self.sec_api_key:
            return {"has_issues": False, "issues": []}
        
        try:
            # Use SEC API to get recent filings
            url = f"https://api.sec-api.io?token={self.sec_api_key}"
            # This is a placeholder - actual implementation would use proper SEC API endpoints
            
            # For now, return mock data
            return {
                "has_issues": False,
                "issues": [],
                "last_filing": "2023-10-15"
            }
        except Exception as e:
            st.warning(f"SEC API check failed: {e}")
            return {"has_issues": False, "issues": []}
    
    def analyze_legal_risks(self, ticker: str, company_name: str, news_items: List[Dict]) -> Dict[str, Any]:
        """Analyze legal risks for a company using both SEC data and news analysis"""
        sec_data = self.check_sec_filings(ticker)
        
        # Analyze news for legal risks
        legal_news = []
        risk_score = 0
        
        legal_risk_keywords = {
            "lawsuit", "litigation", "settlement", "investigation", "probe", "subpoena",
            "violation", "fine", "penalty", "sanction", "compliance", "regulation",
            "ftc", "sec", "doj", "justice department", "antitrust", "monopoly",
            "breach", "fraud", "misconduct", "whistleblower", "subpoena", "enforcement"
        }
        
        for news in news_items:
            content = f"{news.get('title', '')} {news.get('description', '')}".lower()
            risk_keywords = [kw for kw in legal_risk_keywords if kw in content]
            
            if risk_keywords:
                risk_score += len(risk_keywords) * 0.1
                legal_news.append({
                    "title": news.get("title", "")[:100],
                    "keywords": risk_keywords,
                    "source": news.get("source", ""),
                    "date": news.get("publishedAt", "")
                })
        
        # Combine SEC issues and news-based risks
        all_issues = sec_data.get("issues", []) + [f"Legal risk mentioned in news: {news['title']}" for news in legal_news[:3]]
        
        # Normalize risk score (0 to 1)
        risk_score = min(1.0, risk_score)
        
        # Add SEC issues to risk score
        if sec_data.get("has_issues", False):
            risk_score = min(1.0, risk_score + 0.3)
        
        return {
            "risk_score": risk_score,
            "issues": all_issues[:5],  # Top 5 issues
            "legal_news": legal_news[:3]  # Top 3 legal news items
        }

# -------------- Cultural & Employee Sentiment Analysis --------------

class AdvancedCultureAnalyzer:
    """Advanced culture analysis using multiple data sources"""
    def __init__(self):
        self.culture_dimensions = {
            "innovation": {"innovate", "innovation", "research", "develop", "technology", "patent", "create", "breakthrough", "disrupt"},
            "customer_focus": {"customer", "client", "user", "experience", "service", "support", "satisfaction", "delight", "centric"},
            "efficiency": {"efficient", "lean", "optimize", "productivity", "cost", "streamline", "automate", "process", "system"},
            "quality": {"quality", "excellence", "standard", "best", "premium", "superior", "reliable", "craftsmanship", "precision"},
            "collaboration": {"team", "collaborate", "together", "partner", "cooperate", "unity", "share", "synergy", "collective"},
            "ethics": {"ethical", "integrity", "honest", "transparent", "trust", "values", "principle", "moral", "responsible"},
            "agility": {"agile", "adapt", "flexible", "responsive", "nimble", "quick", "pivot", "change", "dynamic"}
        }

    def analyze_culture(self, texts: List[str], news_items: List[Dict] = None) -> Dict[str, Any]:
        """Analyze cultural aspects from provided texts and news"""
        if not texts:
            return {"alignment": 0.5, "features": {}, "compatibility": 0.5}
            
        combined_text = " ".join(texts).lower()
        features = {}
        
        # Analyze each cultural dimension
        for category, keywords in self.culture_dimensions.items():
            count = sum(1 for keyword in keywords if keyword in combined_text)
            # Normalize based on text length and keyword set size
            features[category] = min(1.0, count / (len(keywords) * 0.3))
        
        # Calculate overall cultural score (weighted average)
        weights = {
            "innovation": 0.15,
            "customer_focus": 0.20,
            "efficiency": 0.15,
            "quality": 0.15,
            "collaboration": 0.15,
            "ethics": 0.10,
            "agility": 0.10
        }
        
        alignment = sum(features[dim] * weight for dim, weight in weights.items() if dim in features)
        
        # Analyze news for cultural mentions
        culture_news = []
        if news_items:
            for news in news_items:
                content = f"{news.get('title', '')} {news.get('description', '')}".lower()
                for category in self.culture_dimensions.keys():
                    if any(keyword in content for keyword in self.culture_dimensions[category]):
                        culture_news.append({
                            "title": news.get("title", "")[:100],
                            "category": category,
                            "source": news.get("source", "")
                        })
        
        return {
            "alignment": alignment,
            "features": features,
            "culture_news": culture_news[:5],
            "compatibility": 0.5  # Will be calculated when comparing two companies
        }

    def calculate_cultural_compatibility(self, culture_a: Dict, culture_b: Dict) -> float:
        """Calculate how compatible two company cultures are"""
        if not culture_a or not culture_b:
            return 0.5
        
        # Calculate similarity for each dimension
        similarities = []
        for dimension in self.culture_dimensions.keys():
            if dimension in culture_a["features"] and dimension in culture_b["features"]:
                # Similarity is 1 - absolute difference
                similarity = 1 - abs(culture_a["features"][dimension] - culture_b["features"][dimension])
                similarities.append(similarity)
        
        # Calculate weighted average based on importance of each dimension
        weights = {
            "innovation": 0.15,
            "customer_focus": 0.20,
            "efficiency": 0.15,
            "quality": 0.15,
            "collaboration": 0.15,
            "ethics": 0.10,
            "agility": 0.10
        }
        
        if similarities:
            total_weight = sum(weights.get(dim, 0) for dim in self.culture_dimensions.keys())
            weighted_sum = sum(similarities[i] * list(weights.values())[i] for i in range(min(len(similarities), len(weights))))
            return weighted_sum / total_weight
        
        return 0.5

# -------------- Strategic Fit Analysis --------------

class StrategicAnalyzer:
    """Analyze strategic fit between companies"""
    def __init__(self):
        self.industry_synergies = {
            "Technology": {"Technology": 0.8, "Healthcare": 0.6, "Financial Services": 0.7, "Consumer Cyclical": 0.5, "Industrials": 0.4},
            "Healthcare": {"Technology": 0.6, "Healthcare": 0.9, "Financial Services": 0.4, "Consumer Cyclical": 0.3, "Industrials": 0.5},
            "Financial Services": {"Technology": 0.7, "Healthcare": 0.4, "Financial Services": 0.8, "Consumer Cyclical": 0.6, "Industrials": 0.4},
            "Consumer Cyclical": {"Technology": 0.5, "Healthcare": 0.3, "Financial Services": 0.6, "Consumer Cyclical": 0.7, "Industrials": 0.5},
            "Industrials": {"Technology": 0.4, "Healthcare": 0.5, "Financial Services": 0.4, "Consumer Cyclical": 0.5, "Industrials": 0.8}
        }
    
    def analyze_strategic_fit(self, financial_a: FinancialSnapshot, financial_b: FinancialSnapshot, 
                             transaction_type: str) -> Dict[str, Any]:
        """Analyze strategic fit between two companies"""
        sector_a = financial_a.sector or "Unknown"
        sector_b = financial_b.sector or "Unknown"
        
        # Calculate industry synergy
        industry_synergy = self.industry_synergies.get(sector_a, {}).get(sector_b, 0.5)
        
        # Calculate market position compatibility
        market_cap_ratio = 0.5
        if financial_a.market_cap and financial_b.market_cap:
            ratio = min(financial_a.market_cap, financial_b.market_cap) / max(financial_a.market_cap, financial_b.market_cap)
            if transaction_type == "Acquisition":
                # For acquisitions, ideal ratio is smaller target (0.1-0.3)
                if ratio <= 0.3:
                    market_cap_ratio = 0.8
                elif ratio <= 0.5:
                    market_cap_ratio = 0.6
                else:
                    market_cap_ratio = 0.4
            else:
                # For mergers, more similar size is better (0.5-0.8)
                if ratio >= 0.7:
                    market_cap_ratio = 0.8
                elif ratio >= 0.5:
                    market_cap_ratio = 0.6
                else:
                    market_cap_ratio = 0.4
        
        # Calculate geographic fit (placeholder - would use real data)
        geographic_fit = 0.6
        
        # Calculate product/service complementarity
        complementarity = 0.5  # Placeholder
        
        # Weighted average of factors
        weights = [0.3, 0.25, 0.2, 0.25]  # industry, market cap, geographic, complementarity
        factors = [industry_synergy, market_cap_ratio, geographic_fit, complementarity]
        
        strategic_score = sum(weight * factor for weight, factor in zip(weights, factors))
        
        return {
            "score": strategic_score,
            "industry_synergy": industry_synergy,
            "market_position": market_cap_ratio,
            "geographic_fit": geographic_fit,
            "complementarity": complementarity,
            "sector_a": sector_a,
            "sector_b": sector_b
        }

# -------------- Recommendation Engine --------------

class AdvancedRecommendationEngine:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "financial": 0.25,
            "strategic": 0.20,
            "cultural": 0.15,
            "sentiment": 0.15,
            "legal": 0.10,
            "synergy": 0.15
        }

    def calculate_score(self, financial: float, strategic: float, cultural: float, 
                       sentiment: float, legal: float, synergy: float) -> float:
        score = 0.0
        score += self.weights["financial"] * financial
        score += self.weights["strategic"] * strategic
        score += self.weights["cultural"] * cultural
        score += self.weights["sentiment"] * sentiment
        score += self.weights["legal"] * (1 - legal)  # Legal risk is inverted
        score += self.weights["synergy"] * synergy
        return min(1.0, max(0.0, score))

    def make_recommendation(self, score: float, transaction_type: str) -> Tuple[str, str, List[str]]:
        if score > 0.8:
            recommendation = "GO"
            rationale = f"Strong recommendation to proceed with the {transaction_type.lower()} (Score: {score:.2f}/1.0)"
            actions = [
                "Proceed with formal due diligence",
                "Engage legal and financial advisors",
                "Develop detailed integration plan",
                "Begin stakeholder communications"
            ]

        elif score <= 0.8 and score >= 0.7:
            recommendation = "GO with Partial Integration"
            rationale = f"Recommend proceeding with caution for the {transaction_type.lower()} (Score: {score:.2f}/1.0)"
            actions = [
                "Address specific risk areas identified",
                "Enhance due diligence in weak areas",
                "Develop robust risk mitigation strategies",
                "Engage cultural integration experts"
            ]
            
        elif score < 0.7 and score >= 0.6:
            recommendation = "HOLD"
            rationale = f"Proceed with caution and further due diligence for the {transaction_type.lower()} (Score: {score:.2f}/1.0)"
            actions = [
                "Conduct additional market research",
                "Perform deeper financial analysis",
                "Address identified risk factors",
                "Consider pilot collaboration before full transaction"
            ]
        else:
            recommendation = "NO-GO"
            rationale = f"Recommend against proceeding with the {transaction_type.lower()} (Score: {score:.2f}/1.0)"
            actions = [
                "Consider alternative targets or strategies",
                "If strategic fit is high, consider minority investment instead",
                "Monitor target for improvements in risk areas",
                "Reassess in 6-12 months"
            ]
        
        return recommendation, rationale, actions

# -------------- Strategy Generator --------------

class AdvancedStrategyGenerator:
    def __init__(self):
        self.groq_api_key = GROQ_API_KEY

    def generate_strategy(self, company_a: str, company_b: str, transaction_type: str, 
                         scores: Dict, financial_a: FinancialSnapshot, financial_b: FinancialSnapshot,
                         news_analysis: Dict, legal_analysis: Dict, cultural_analysis: Dict) -> str:
        
        # Format market cap for display
        market_cap_a = f"{financial_a.market_cap:,.0f}" if financial_a.market_cap else "N/A"
        market_cap_b = f"{financial_b.market_cap:,.0f}" if financial_b.market_cap else "N/A"
        
        prompt = f"""
Create a comprehensive {transaction_type.lower()} integration strategy for {company_a} and {company_b}.

COMPANY PROFILES:
- {company_a}: {financial_a.sector or 'Unknown'} sector, Market Cap: ${market_cap_a}
- {company_b}: {financial_b.sector or 'Unknown'} sector, Market Cap: ${market_cap_b}

KEY ASSESSMENT SCORES:
- Financial Compatibility: {scores.get('financial', 0):.2f}/1.0
- Strategic Fit: {scores.get('strategic', 0):.2f}/1.0
- Cultural Alignment: {scores.get('cultural', 0):.2f}/1.0
- Market Sentiment: {scores.get('sentiment', 0):.2f}/1.0
- Legal Risk: {scores.get('legal', 0):.2f}/1.0
- Synergy Potential: {scores.get('synergy', 0):.2f}/1.0

RISK FACTORS:
{json.dumps(news_analysis.get('risk_factors', []) + legal_analysis.get('issues', []), indent=2)}

Please provide a detailed integration strategy covering:
1. Strategic rationale and synergy opportunities specific to these companies
2. Organizational integration approach addressing the cultural alignment score
3. Technology and systems integration plan
4. Cultural integration plan with specific initiatives
5. Risk mitigation strategies for the identified risks
6. Timeline and key milestones for the integration
7. Financial considerations based on the companies' profiles
8. Communication strategy for stakeholders
Give only 2 bullet point per section to keep it concise.

The strategy should be practical, actionable, and tailored to these specific companies.
"""

        if self.groq_api_key:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": "You are an expert M&A integration strategist with deep industry knowledge. Provide detailed, actionable advice."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }

                response = requests.post(url, headers=headers, json=payload, timeout=45)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                return self._generate_fallback_strategy(company_a, company_b, transaction_type, scores)
        else:
            return self._generate_fallback_strategy(company_a, company_b, transaction_type, scores)

    def _generate_fallback_strategy(self, company_a: str, company_b: str, 
                                  transaction_type: str, scores: Dict) -> str:
        return f"""
# Integration Strategy: {company_a} → {company_b} ({transaction_type})

## 📊 Assessment Summary
- **Financial Compatibility**: {scores.get('financial', 0):.2f}/1.0
- **Strategic Fit**: {scores.get('strategic', 0):.2f}/1.0
- **Cultural Alignment**: {scores.get('cultural', 0):.2f}/1.0
- **Market Sentiment**: {scores.get('sentiment', 0):.2f}/1.0
- **Legal Risk**: {scores.get('legal', 0):.2f}/1.0

## 🎯 Strategic Rationale
Potential synergies in {'same sector' if 'sector_a' in scores and 'sector_b' in scores and scores['sector_a'] == scores['sector_b'] else 'cross-sector'} expansion and technology integration through this {transaction_type.lower()}.

## 📅 Recommended Approach
1. **Phase 1 (0-30 days)**: Due diligence completion and initial planning
2. **Phase 2 (30-90 days)**: Operational integration and team alignment  
3. **Phase 3 (90-180 days)**: Full integration and synergy realization

## ⚠️ Risk Considerations
- Monitor cultural integration closely (Alignment: {scores.get('cultural', 0):.2f}/1.0)
- Ensure clear communication throughout the process
- Develop contingency plans for key risk areas

*Enable GROQ API for AI-generated detailed strategy*
"""

# -------------- Visualization Utilities --------------

class VisualizationHelper:
    """Helper class for creating visualizations"""
    @staticmethod
    def create_radar_chart(scores: Dict, title: str) -> go.Figure:
        categories = list(scores.keys())
        values = list(scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the circle
            theta=categories + [categories[0]],
            fill='toself',
            name=title
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title=title
        )
        
        return fig

    @staticmethod
    def create_score_comparison(company_a: str, company_b: str, financial_a: Dict, financial_b: Dict) -> go.Figure:
        metrics = ['Market Cap', 'P/E Ratio', 'Profit Margin', 'Debt-to-Equity']
        values_a = []
        values_b = []
        
        for metric in metrics:
            if metric == 'Market Cap':
                val_a = financial_a.get('market_cap', 0)
                val_b = financial_b.get('market_cap', 0)
                # Normalize for comparison
                max_val = max(val_a, val_b) or 1
                values_a.append((val_a or 0) / max_val)
                values_b.append((val_b or 0) / max_val)
            elif metric == 'P/E Ratio':
                val_a = financial_a.get('pe', 0)
                val_b = financial_b.get('pe', 0)
                max_val = max(val_a, val_b, 1) or 1
                values_a.append((val_a or 0) / max_val)
                values_b.append((val_b or 0) / max_val)
            elif metric == 'Profit Margin':
                val_a = financial_a.get('profit_margin', 0) or 0
                val_b = financial_b.get('profit_margin', 0) or 0
                values_a.append(val_a)
                values_b.append(val_b)
            elif metric == 'Debt-to-Equity':
                val_a = financial_a.get('debt_to_equity', 0) or 0
                val_b = financial_b.get('debt_to_equity', 0) or 0
                # Inverse since lower is better
                max_val = max(val_a, val_b, 1) or 1
                values_a.append(1 - (val_a / max_val))
                values_b.append(1 - (val_b / max_val))
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=company_a,
            x=metrics,
            y=values_a,
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            name=company_b,
            x=metrics,
            y=values_b,
            marker_color='orange'
        ))
        
        fig.update_layout(
            title="Financial Metrics Comparison (Normalized)",
            barmode='group',
            yaxis=dict(range=[0, 1])
        )
        
        return fig

# -------------- Main Application --------------

def main():
    st.set_page_config(page_title="M&A Due Diligence Analyzer", layout="wide", page_icon="🤝")
    st.title("🤝 M&A Due Diligence Analysis Platform")
    
    # Initialize session state
    if "analysis_run" not in st.session_state:
        st.session_state.analysis_run = False
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Transaction type selection
        transaction_type = st.radio(
            "Transaction Type",
            ["Acquisition", "Merger"],
            help="Select whether this is an acquisition (one company buying another) or a merger (two companies combining)"
        )
        
        st.subheader("Company A (Acquirer)" if transaction_type == "Acquisition" else "Company A (Merging Party)")
        company_a = st.text_input("Company A Name", "Microsoft", key="company_a")
        ticker_a = st.text_input("Company A Ticker", "MSFT", key="ticker_a").upper()
        company_a_desc = st.text_area("Company A Description", "Technology company focused on software, cloud computing, and enterprise solutions", key="desc_a")
        
        st.subheader("Company B (Target)" if transaction_type == "Acquisition" else "Company B (Merging Party)")  
        company_b = st.text_input("Company B Name", "Adobe", key="company_b")
        ticker_b = st.text_input("Company B Ticker", "ADBE", key="ticker_b").upper()
        company_b_desc = st.text_area("Company B Description", "Creative software company known for Photoshop and digital experience tools", key="desc_b")
        
        st.markdown("---")
        st.subheader("📄 Due Diligence Documents (Optional)")
        uploaded_files = st.file_uploader("Upload financial reports, company profiles, or other relevant documents", 
                                        type=["pdf", "txt", "md", "csv", "xlsx", "xls"],
                                        accept_multiple_files=True,
                                        help="Upload any available documents for deeper analysis")
        
        st.markdown("---")
        st.subheader("🔍 API Configuration")
        if st.button("Check API Status", help="Verify API connections"):
            with st.spinner("Checking APIs..."):
                st.write("📰 NewsAPI:", "✅ Configured" if NEWSAPI_KEY else "❌ Not configured")
                st.write("🤖 GROQ API:", "✅ Configured" if GROQ_API_KEY else "❌ Not configured")
                st.write("📊 Alpha Vantage:", "✅ Configured" if ALPHA_VANTAGE_KEY else "❌ Not configured")
                st.write("⚖️ SEC API:", "✅ Configured" if SEC_API_KEY else "❌ Not configured")
                
                if not NEWSAPI_KEY:
                    st.info("NewsAPI key not configured. Some news analysis features will be limited.")
                if not GROQ_API_KEY:
                    st.info("GROQ API key not configured. AI-generated strategy will be limited.")
                if not ALPHA_VANTAGE_KEY:
                    st.info("Alpha Vantage key not configured. Some financial data may be limited.")
                if not SEC_API_KEY:
                    st.info("SEC API key not configured. Some legal analysis features will be limited.")

    # Main analysis button
    if st.button("🚀 Run Comprehensive Due Diligence Analysis", type="primary", use_container_width=True):
        with st.spinner("Conducting comprehensive due diligence analysis..."):
            st.session_state.analysis_run = True
            run_analysis(company_a, ticker_a, company_a_desc, 
                        company_b, ticker_b, company_b_desc, uploaded_files, transaction_type)

    # Show placeholder before analysis is run
    if not st.session_state.analysis_run:
        st.info("👈 Configure companies and click 'Run Analysis' to begin due diligence")
        
        # Show sample dashboard
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Financial Score", "0.72", "0.12")
        with col2:
            st.metric("Cultural Alignment", "0.68", "-0.05")
        with col3:
            st.metric("Risk Assessment", "0.35", "0.08")
        
        st.write("This tool analyzes financial compatibility, cultural alignment, market sentiment, and legal risks to provide a comprehensive due diligence assessment for M&A transactions.")

def run_analysis(company_a, ticker_a, company_a_desc, company_b, ticker_b, company_b_desc, uploaded_files, transaction_type):
    # Initialize analyzers
    finance_fetcher = AdvancedFinanceFetcher()
    news_fetcher = AdvancedNewsFetcher()
    legal_analyzer = LegalAnalyzer()
    culture_analyzer = AdvancedCultureAnalyzer()
    strategic_analyzer = StrategicAnalyzer()
    recommender = AdvancedRecommendationEngine()
    strategy_gen = AdvancedStrategyGenerator()
    doc_ingestor = DocIngestor()
    visualizer = VisualizationHelper()
    
    # Extract data from uploaded documents
    doc_texts = []
    doc_financial_data = {"a": {}, "b": {}}
    
    if uploaded_files:
        st.header("📄 Document Analysis")
        for file in uploaded_files:
            text = doc_ingestor.parse_file(file, file.name)
            if text:
                doc_texts.append(text)
                # Try to extract financial data
                financial_data = doc_ingestor.extract_financial_data(text)
                if financial_data:
                    # Try to determine which company this document relates to
                    if company_a.lower() in text.lower():
                        doc_financial_data["a"].update(financial_data)
                    elif company_b.lower() in text.lower():
                        doc_financial_data["b"].update(financial_data)
        
        if doc_texts:
            st.success(f"Processed {len(uploaded_files)} documents with {sum(len(text) for text in doc_texts):,} characters")
        else:
            st.info("No extractable text found in uploaded documents")
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Financial Analysis", 
        "📰 News & Sentiment", 
        "🏢 Cultural Alignment", 
        "🎯 Strategic Fit",
        "⚖️ Legal & Risk", 
        "🎯 Recommendation", 
        "📋 Strategy"
    ])
    
    # 1. Financial Analysis
    with tab1:
        st.header("💰 Financial Analysis")
        
        with st.spinner("Fetching financial data..."):
            financial_a = finance_fetcher.fetch(ticker_a)
            financial_b = finance_fetcher.fetch(ticker_b)
        
        # Enhance with document data if available
        if doc_financial_data["a"]:
            for key, value in doc_financial_data["a"].items():
                if not getattr(financial_a, key, None):
                    setattr(financial_a, key, value)
        
        if doc_financial_data["b"]:
            for key, value in doc_financial_data["b"].items():
                if not getattr(financial_b, key, None):
                    setattr(financial_b, key, value)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{company_a} ({ticker_a})")
            if financial_a.price:
                st.metric("Current Price", f"${financial_a.price:,.2f}")
            if financial_a.market_cap:
                st.metric("Market Cap", f"${financial_a.market_cap:,.0f}")
            if financial_a.pe:
                st.metric("P/E Ratio", f"{financial_a.pe:.2f}")
            if financial_a.debt_to_equity:
                st.metric("Debt-to-Equity", f"{financial_a.debt_to_equity:.2f}")
            if financial_a.profit_margin is not None:
                st.metric("Profit Margin", f"{financial_a.profit_margin:.2%}")
            if financial_a.sector:
                st.metric("Sector", financial_a.sector)
        
        with col2:
            st.subheader(f"{company_b} ({ticker_b})")
            if financial_b.price:
                st.metric("Current Price", f"${financial_b.price:,.2f}")
            if financial_b.market_cap:
                st.metric("Market Cap", f"${financial_b.market_cap:,.0f}")
            if financial_b.pe:
                st.metric("P/E Ratio", f"{financial_b.pe:.2f}")
            if financial_b.debt_to_equity:
                st.metric("Debt-to-Equity", f"{financial_b.debt_to_equity:.2f}")
            if financial_b.profit_margin is not None:
                st.metric("Profit Margin", f"{financial_b.profit_margin:.2%}")
            if financial_b.sector:
                st.metric("Sector", financial_b.sector)
        
        # Calculate financial score
        financial_score = finance_fetcher.calculate_financial_score(financial_a, financial_b, transaction_type)
        st.metric("Financial Compatibility Score", f"{financial_score:.2f}/1.0")
        
        with st.expander("📈 Financial Score Details"):
            st.write("""
            This score evaluates how well the companies' financial profiles align for a successful transaction:
            - **Market Cap Compatibility**: Compares the relative sizes of the companies
            - **Valuation Metrics**: Analyzes P/E ratios and other valuation measures
            - **Profitability**: Compares profit margins and financial health
            - **Leverage**: Examines debt levels and financial risk
            - **Growth Patterns**: Assesses revenue and earnings growth compatibility
            """)
            
            if financial_score >= 0.7:
                st.success("High financial compatibility suggests strong foundation for transaction.")
            elif financial_score >= 0.5:
                st.warning("Moderate financial compatibility - further analysis recommended.")
            else:
                st.error("Low financial compatibility may present significant challenges.")
    
    # 2. News & Sentiment Analysis
    with tab2:
        st.header("📰 Market News & Sentiment")
        
        # Fetch and analyze news for both companies
        with st.spinner("Gathering relevant news..."):
            news_items_a = news_fetcher.fetch_news(company_a, transaction_type)
            news_items_b = news_fetcher.fetch_news(company_b, transaction_type)
        
        # Combine news items
        all_news_items = news_items_a + news_items_b
        
        # Analyze sentiment
        news_analysis = news_fetcher.analyze_news_sentiment(all_news_items)
        sentiment_score = news_analysis.get("score", 0.5)
        
        # Display news
        tab2_1, tab2_2 = st.tabs(["News Analysis", "News Articles"])
        
        with tab2_1:
            st.metric("Overall Sentiment Score", f"{sentiment_score:.2f}/1.0")
            
            if news_analysis.get("risk_factors"):
                st.subheader("⚠️ Identified Risk Factors")
                for risk in news_analysis["risk_factors"][:3]:  # Show top 3
                    st.warning(risk)
            
            if news_analysis.get("positive_factors"):
                st.subheader("✅ Positive Factors")
                for positive in news_analysis["positive_factors"][:3]:  # Show top 3
                    st.success(positive)
        
        with tab2_2:
            for i, news in enumerate(all_news_items[:10]):  # Show first 10 articles
                with st.expander(f"{i+1}. {news.get('title', 'No title')}"):
                    st.write(f"**Source:** {news.get('source', 'Unknown')}")
                    st.write(f"**Date:** {news.get('publishedAt', 'Unknown')}")
                    st.write(news.get('description', 'No description available'))
                    if news.get('url') and news.get('url') != "#":
                        st.write(f"[Read full article]({news.get('url')})")
    
    # 3. Culture Analysis
    with tab3:
        st.header("🏢 Cultural Alignment Analysis")
        
        # Prepare texts for cultural analysis (company descriptions + document text)
        culture_texts_a = [company_a_desc]
        culture_texts_b = [company_b_desc]
        
        # Add document text if related to each company
        for text in doc_texts:
            if company_a.lower() in text.lower():
                culture_texts_a.append(text)
            if company_b.lower() in text.lower():
                culture_texts_b.append(text)
        
        with st.spinner("Analyzing cultural alignment..."):
            culture_a = culture_analyzer.analyze_culture(culture_texts_a, news_items_a)
            culture_b = culture_analyzer.analyze_culture(culture_texts_b, news_items_b)
        
        cultural_compatibility = culture_analyzer.calculate_cultural_compatibility(culture_a, culture_b)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{company_a} Culture Score", f"{culture_a.get('alignment', 0.5):.2f}/1.0")
        
        with col2:
            st.metric(f"{company_b} Culture Score", f"{culture_b.get('alignment', 0.5):.2f}/1.0")
        
        with col3:
            st.metric("Cultural Compatibility", f"{cultural_compatibility:.2f}/1.0")
        
        # Show cultural dimensions
        with st.expander("🔍 Cultural Dimensions Analysis"):
            st.subheader(f"{company_a} Cultural Profile")
            for dimension, score in culture_a.get("features", {}).items():
                st.progress(score, text=f"{dimension.replace('_', ' ').title()}: {score:.2f}")
            
            st.subheader(f"{company_b} Cultural Profile")
            for dimension, score in culture_b.get("features", {}).items():
                st.progress(score, text=f"{dimension.replace('_', ' ').title()}: {score:.2f}")
    
    # 4. Strategic Fit Analysis
    with tab4:
        st.header("🎯 Strategic Fit Analysis")
        
        with st.spinner("Analyzing strategic fit..."):
            strategic_fit = strategic_analyzer.analyze_strategic_fit(financial_a, financial_b, transaction_type)
            strategic_score = strategic_fit.get("score", 0.5)
        
        st.metric("Strategic Fit Score", f"{strategic_score:.2f}/1.0")
        
        with st.expander("📊 Strategic Fit Details"):
            st.write(f"**Industry Synergy**: {strategic_fit.get('industry_synergy', 0.5):.2f}/1.0")
            st.write(f"**Market Position Compatibility**: {strategic_fit.get('market_position', 0.5):.2f}/1.0")
            st.write(f"**Geographic Fit**: {strategic_fit.get('geographic_fit', 0.5):.2f}/1.0")
            st.write(f"**Product/Service Complementarity**: {strategic_fit.get('complementarity', 0.5):.2f}/1.0")
            st.write(f"**Sector Analysis**: {strategic_fit.get('sector_a', 'Unknown')} → {strategic_fit.get('sector_b', 'Unknown')}")
    
    # 5. Legal & Risk Analysis
    with tab5:
        st.header("⚖️ Legal & Risk Assessment")
        
        with st.spinner("Analyzing legal risks..."):
            legal_analysis_a = legal_analyzer.analyze_legal_risks(ticker_a, company_a, news_items_a)
            legal_analysis_b = legal_analyzer.analyze_legal_risks(ticker_b, company_b, news_items_b)
        
        # Use the higher risk score (worst case)
        legal_risk_score = max(legal_analysis_a.get("risk_score", 0.2), legal_analysis_b.get("risk_score", 0.2))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(f"{company_a} Legal Risk", f"{legal_analysis_a.get('risk_score', 0.2):.2f}/1.0")
        
        with col2:
            st.metric(f"{company_b} Legal Risk", f"{legal_analysis_b.get('risk_score', 0.2):.2f}/1.0")
        
        st.metric("Combined Legal Risk Score", f"{legal_risk_score:.2f}/1.0")
        
        # Show legal issues if any
        all_issues = legal_analysis_a.get("issues", []) + legal_analysis_b.get("issues", [])
        if all_issues:
            with st.expander("📋 Legal Issues Details"):
                for issue in all_issues:
                    st.error(issue)
        else:
            st.success("No significant legal issues identified")
    
    # 6. Synergy Potential
    st.header("✨ Synergy Potential")
    
    # Calculate synergy score based on multiple factors
    synergy_score = 0.5  # Base score
    
    # Adjust based on strategic fit
    synergy_score += (strategic_score - 0.5) * 0.3
    
    # Adjust based on cultural compatibility
    synergy_score += (cultural_compatibility - 0.5) * 0.2
    
    # Adjust based on financial compatibility
    synergy_score += (financial_score - 0.5) * 0.2
    
    # Cap between 0 and 1
    synergy_score = max(0, min(1, synergy_score))
    
    st.metric("Synergy Potential Score", f"{synergy_score:.2f}/1.0")
    
    with st.expander("💡 Synergy Opportunities"):
        if strategic_fit.get("industry_synergy", 0.5) > 0.7:
            st.success("**High Industry Synergy**: Companies operate in closely related sectors with significant overlap and complementarity.")
        elif strategic_fit.get("industry_synergy", 0.5) > 0.5:
            st.info("**Moderate Industry Synergy**: Some complementary aspects with potential for cross-selling or operational efficiencies.")
        else:
            st.warning("**Low Industry Synergy**: Companies operate in different sectors - may require more effort to realize synergies.")
        
        if cultural_compatibility > 0.7:
            st.success("**High Cultural Compatibility**: Similar values and work styles should facilitate integration.")
        elif cultural_compatibility > 0.5:
            st.info("**Moderate Cultural Compatibility**: Some cultural differences that will need to be managed during integration.")
        else:
            st.warning("**Low Cultural Compatibility**: Significant cultural differences may create integration challenges.")
    
    # 7. Final Recommendation
    with tab6:
        st.header("🎯 Final Recommendation")
        
        # Calculate overall score
        scores = {
            "financial": financial_score,
            "strategic": strategic_score,
            "cultural": cultural_compatibility,
            "sentiment": sentiment_score,
            "legal": legal_risk_score,
            "synergy": synergy_score
        }
        
        overall_score = recommender.calculate_score(
            financial=financial_score,
            strategic=strategic_score,
            cultural=cultural_compatibility,
            sentiment=sentiment_score,
            legal=legal_risk_score,
            synergy=synergy_score
        )
        
        decision, rationale, actions = recommender.make_recommendation(overall_score, transaction_type)
        
        # Display recommendation with appropriate color
        if decision == "GO":
            st.success(f"## ✅ {decision}")
        elif decision == "GO with Partial Integration":
            st.info(f"## ✅ {decision}")
        elif decision == "HOLD":
            st.warning(f"## ⚠️ {decision}")
        else:
            st.error(f"## ❌ {decision}")
        
        st.write(rationale)
        
        # Display score breakdown
        st.subheader("Score Breakdown")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Financial", f"{scores['financial']:.2f}")
        with col2:
            st.metric("Strategic", f"{scores['strategic']:.2f}")
        with col3:
            st.metric("Culture", f"{scores['cultural']:.2f}")
        with col4:
            st.metric("Sentiment", f"{scores['sentiment']:.2f}")
        with col5:
            st.metric("Legal Risk", f"{scores['legal']:.2f}")
        with col6:
            st.metric("Synergy", f"{scores['synergy']:.2f}")
        
        # Display radar chart
        st.plotly_chart(visualizer.create_radar_chart({
            "Financial": scores['financial'],
            "Strategic": scores['strategic'],
            "Cultural": scores['cultural'],
            "Sentiment": scores['sentiment'],
            "Risk": 1 - scores['legal'],  # Invert risk for visualization
            "Synergy": scores['synergy']
        }, "Due Diligence Assessment"), use_container_width=True)
        
        # Display recommended actions
        st.subheader("Recommended Actions")
        for i, action in enumerate(actions, 1):
            st.write(f"{i}. {action}")
    
    # 8. Integration Strategy
    with tab7:
        st.header("📋 Integration Strategy")
        
        with st.spinner("Generating customized integration strategy..."):
            strategy = strategy_gen.generate_strategy(
                company_a, company_b, transaction_type,
                scores, financial_a, financial_b, news_analysis, 
                legal_analysis_b, culture_b  # Focus on target company for risks/culture
            )
        
        st.markdown(strategy)
    
    # 9. Export Results
    st.header("💾 Export Results")
    
    report_data = {
        "analysis_date": time.strftime("%Y-%m-%d"),
        "transaction_type": transaction_type,
        "companies": {
            "company_a": {
                "name": company_a, 
                "ticker": ticker_a,
                "description": company_a_desc,
                "financials": {k: v for k, v in vars(financial_a).items() if not k.startswith('_') and v is not None}
            },
            "company_b": {
                "name": company_b, 
                "ticker": ticker_b,
                "description": company_b_desc,
                "financials": {k: v for k, v in vars(financial_b).items() if not k.startswith('_') and v is not None}
            }
        },
        "scores": scores,
        "decision": decision,
        "overall_score": overall_score,
        "news_analysis": news_analysis,
        "legal_analysis": {
            "company_a": legal_analysis_a,
            "company_b": legal_analysis_b
        },
        "cultural_analysis": {
            "company_a": culture_a,
            "company_b": culture_b,
            "compatibility": cultural_compatibility
        },
        "strategic_analysis": strategic_fit
    }
    
    # Create download button
    json_report = json.dumps(report_data, indent=2, default=str)
    st.download_button(
        label="Download Full JSON Report",
        data=json_report,
        file_name=f"ma_analysis_{company_a}_{company_b}_{time.strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()