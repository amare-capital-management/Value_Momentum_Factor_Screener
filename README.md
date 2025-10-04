<img width="780" height="253" alt="ACM w color" src="https://github.com/user-attachments/assets/526bc08b-9bbe-426c-994a-fd69ee37089a" />

### The script is a financial analysis tool designed to evaluate and rank stocks based on valuation and momentum factors. Here's a breakdown of its functionality:

*Stock List:* 

Contains a predefined list of stock tickers (primarily South African stocks with .JO suffix).
Valuation Ratios: 
Fetches financial metrics like Price-to-Earnings (P/E), Price-to-Book (P/B), Price-to-Sales (P/S), EV/EBITDA, and EV/GP using Yahoo Finance API.
Handles missing or unavailable data gracefully.
Momentum Analysis:
Computes historical price returns over different periods (1 month, 3 months, 6 months, and 12 months).
Data Cleaning: 
Fills missing values with column averages. Converts percentage strings to numeric values for calculations.
Percentile Ranks: 
Calculates percentile ranks for valuation and momentum metrics. Computes aggregate "Value Score" and "Momentum Score" for each stock.
Ranking: Ranks stocks based on their value and momentum scores.
Output: 
Saves the following CSV files:
stock_valuation_momentum.csv: Combined valuation and momentum data for all stocks.
value_momentum_ranks.csv: Compact view with value and momentum ranks for all stocks.
Sector-specific CSV files in the fundamentals directory.
Concurrency: 
Uses ThreadPoolExecutor to fetch data for multiple stocks in parallel, improving performance.
**Workflow:*** 
The script fetches valuation ratios and historical price data for the listed stocks. It processes and cleans the data, calculates ranks, and generates output files. The main() function orchestrates the entire workflow, and the script runs when executed directly.

### This script is useful for investors or analysts looking to screen stocks based on fundamental and technical factors.
