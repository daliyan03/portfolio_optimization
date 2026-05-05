import streamlit as st
import numpy as np
import cvxpy as cp
import yfinance as yf


st.title("Portfolio Optimization App")

max_weight = 0.325


def min_variance(Sigma):
    #lowest possible variance
    n = Sigma.shape[0]
    w = cp.Variable(n)
    cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)),
               [cp.sum(w) == 1, w >= 0, w <= max_weight]).solve()
    return float(w.value @ Sigma @ w.value)


def max_variance(mu, Sigma):
    #highest possible variance
    n = len(mu)
    w = cp.Variable(n)
    cp.Problem(cp.Maximize(mu @ w),
               [cp.sum(w) == 1, w >= 0, w <= max_weight]).solve()
    return float(w.value @ Sigma @ w.value)


def annual_vol_to_daily_var(pct):
    #volatility to daily variance with formulas: daily_vol= annual_vol/sqrt(252), var = (daily_vol)^2
    return (pct / 100 / np.sqrt(252)) ** 2


#optimizing function
def optimize(mu, Sigma, risk_limit):
    n = len(mu)
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Maximize(mu @ w),
        [cp.quad_form(w, Sigma) <= risk_limit,
         cp.sum(w) == 1, w >= 0, w <= max_weight]
    )
    prob.solve()
    return w.value if prob.status in ["optimal", "optimal_inaccurate"] else None


risk_strategies = {
    "conservative": annual_vol_to_daily_var(12),  
    "balanced":     annual_vol_to_daily_var(18),  
    "aggressive":   annual_vol_to_daily_var(28),  
}


def get_risk_limits(mu, Sigma):
    var_min = min_variance(Sigma)
    var_max = max_variance(mu, Sigma)

    limits = {}
    lines = []

    lines.append(f"**Feasible variance range:** [{var_min:.6f}, {var_max:.6f}]")

    for profile, target in risk_strategies.items():
        if target < var_min:
            limit = var_min
            note = "not in range, used min variance"

        elif target > var_max:
            limit = var_max
            note = "not in range, used max variance"

        else:
            limit = target
            note = "in range"

        limits[profile] = limit
        lines.append(f"{profile}: {limit:.6f} ({note})")

    st.markdown("<br>".join(lines), unsafe_allow_html=True)

    return limits


ticker_input = st.text_input(
    "Enter tickers separated by commas:",
    "AAPL, MSFT, GOOGL, AMZN, KO, TLT, GLD"
)

tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]


if st.button("Run optimization"):

    data = yf.download(tickers, period="10y", auto_adjust=True, threads=False)
    returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
    mu    = returns.mean().values
    Sigma = returns.cov().values

    #running the code
    risk_limits = get_risk_limits(mu, Sigma)

    for profile, limit in risk_limits.items():
        weights = optimize(mu, Sigma, limit)
        if weights is None:
            st.write(f"{profile.upper()}: no solution")
            continue

        ret = mu @ weights
        var = weights @ Sigma @ weights

        included = []
        not_included = []

        for ticker, w in sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True):
            if w > 0.0001:
                included.append((ticker, w))
            else:
                not_included.append(ticker)

        lines = []

        lines.append(f"**{profile.upper()} — Return: {ret*252:.2%} | Volatility: {np.sqrt(var*252):.2%}**")

        for ticker, w in included:
            lines.append(f"{ticker}: {w:.2%}")

        if not_included:
            lines.append(f"*Not included:* {', '.join(not_included)}")

        st.markdown("<br>".join(lines), unsafe_allow_html=True)