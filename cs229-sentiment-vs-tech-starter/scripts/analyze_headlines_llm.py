import pandas as pd
import os
import sys
from openai import OpenAI
from pathlib import Path
import json
import time
from tqdm import tqdm
import numpy as np

def load_stock_context(ticker, date_str, data_dir):
    """
    Load OHLCV data for a specific ticker around a date.
    Returns a string summarizing the price action.
    """
    try:
        stock_file = data_dir / f"ohlcv_{ticker}.csv"
        if not stock_file.exists():
            return "No stock price data available."
            
        df = pd.read_csv(stock_file)
        df['date'] = pd.to_datetime(df['date'])
        target_date = pd.to_datetime(date_str)
        
        # Get window: 2 days before to 5 days after
        start_date = target_date - pd.Timedelta(days=5) # Look back a bit more to find recent closes
        end_date = target_date + pd.Timedelta(days=5)
        
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        window_df = df.loc[mask].sort_values('date')
        
        if window_df.empty:
            return "No stock price data found for this date range."
            
        # Format context string
        context = "Stock Price Context (OHLCV):\n"
        context += "Date | Close | % Change (from prev close) | Volume\n"
        
        prev_close = None
        
        # Find close before window to calc change for first row
        pre_window = df[df['date'] < window_df.iloc[0]['date']]
        if not pre_window.empty:
            prev_close = pre_window.iloc[-1]['close']
            
        for _, row in window_df.iterrows():
            date_fmt = row['date'].strftime('%Y-%m-%d')
            close = row['close']
            volume = row['volume']
            
            if prev_close:
                pct_change = ((close - prev_close) / prev_close) * 100
                change_str = f"{pct_change:+.2f}%"
            else:
                change_str = "N/A"
            
            is_target = " (Headline Date)" if row['date'].date() == target_date.date() else ""
            context += f"{date_fmt} | ${close:.2f} | {change_str} | {volume}{is_target}\n"
            prev_close = close
            
        return context
    except Exception as e:
        return f"Error loading stock data: {str(e)}"

def analyze_headline_single(client, headline_row, stock_context, model="deepseek-chat"):
    """
    Analyze a single headline with stock context.
    """
    
    system_prompt = """You are a financial expert. Your task is to analyze stock news headlines and determine their potential importance regarding stock price movement.

Output a JSON object with:
1. "score": A float between 0.0 and 1.0 representing the importance/impact potential (0=irrelevant/noise, 1=critical/market-moving).
2. "reasoning": A concise sentence explaining why, referencing the price action if relevant.

Factors:
- High impact: Earnings surprises, M&A, major product launches, regulatory rulings.
- Medium impact: Analyst ratings, minor product news.
- Low impact: General fluff, minor rumors, known events.
"""

    user_content = f"""Headline Analysis:
Date: {headline_row['date']}
Ticker: {headline_row['ticker']}
Headline: {headline_row['text']}
Source: {headline_row['source']}

{stock_context}

Evaluate the importance of this headline."""

    retries = 3
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                response_format={ "type": "json_object" }
            )
            
            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            # Normalize output
            if isinstance(parsed, list):
                parsed = parsed[0] # Should be single object
            
            return parsed

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"Error analyzing headline: {e}")
            return None

def main():
    # Check for API Key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not found.")
        print("Please run: export DEEPSEEK_API_KEY='your_key_here'")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    headlines_file = data_dir / "news" / "headlines.csv"
    output_file = project_root / "data" / "processed" / "headlines_llm_scores.csv"

    print(f"Loading headlines from {headlines_file}")
    df = pd.read_csv(headlines_file)
    
    # Cache stock data contexts to avoid repeated disk reads for same ticker/date? 
    # Since sorted by date usually, maybe not needed, but good for performance.
    # But loading full file is fast enough.
    
    print(f"Processing {len(df)} headlines one by one...")
    
    llm_scores = []
    llm_reasoning = []
    
    # Use tqdm for progress bar
    for i, row in tqdm(df.iterrows(), total=len(df)):

        context = load_stock_context(row['ticker'], row['date'], data_dir)
        
        res = analyze_headline_single(client, row, context)
        if res and isinstance(res, dict):
            llm_scores.append(res.get('score', 0.0))
            llm_reasoning.append(res.get('reasoning', ''))
        else:
            llm_scores.append(0.0)
            llm_reasoning.append("Error processing this headline")
            
        # Rate limit protection (DeepSeek might have limits)
        time.sleep(0.1)
        
        # Periodically save progress
        if (i + 1) % 10 == 0:
            df_partial = df.iloc[:len(llm_scores)].copy()
            df_partial['llm_importance_score'] = llm_scores
            df_partial['llm_reasoning'] = llm_reasoning
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df_partial.to_csv(output_file, index=False)

    # Final assignment
    df['llm_importance_score'] = llm_scores
    df['llm_reasoning'] = llm_reasoning
    
    # Final Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved final results to {output_file}")

if __name__ == "__main__":
    main()
