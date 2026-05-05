# Mean-Variance Portfolio Optimizer

This project builds an optimized stock portfolio by maximizing return while controlling risk.

**Motivation:** applying concepts from convex optimization to a real-world finance problem.  

**Built with:** Python, cvxpy, yfinance, numpy, and Streamlit

---

## Running the code

```bash
git clone https://github.com/daliyan03/portfolio_optimization.git
cd portfolio_optimization
python -m venv venv
```
Activate the environment:

Windows: venv\Scripts\activate  
Mac/Linux: source venv/bin/activate

Install dependencies and run the app:
```bash
pip install -r requirements.txt
streamlit run code.py
```
## Optimization Problem
max Σ wᵢ rᵢ  
Σ wᵢ = 1  
wᵀ Σ w ≤ Rmax  
0 ≤ wᵢ ≤ 0.325

## Risk Profiles:  
Conservative: 12%  
Balanced: 18%  
Aggressive: 28%  

## Key Findings: 
Diversification improves returns under a risk constraint.
Mixed portfolios achieved higher returns than high-return-only portfolios due to lower correlation between assets.

Disclaimer:  
Returns are based on historical data and may not reflect future performance. 

## Screenshots

  <img src="images/results.png" width="500"><br>
  <img src="images/results2.png" width="500"><br>
  <em>Example run </em>
</div>