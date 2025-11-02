import argparse  # Used to read command-line arguments: --ticker, --years, etc.
import numpy as np  # Numerical math operations (arrays, IRR, NPV discounting)
import pandas as pd  # For DataFrames and data manipulation
import matplotlib.pyplot as plt  # For creating charts/graphs
from dataclasses import dataclass  # Handy for storing scenario settings as a class
from typing import Optional, Dict  # For cleaner type hints

import yfinance as yf  # Finance API for stock ticker data


# Define a data structure
@dataclass
class Scenario:
    cogs_improvement_pp: float = 1.0   # How many percentage points COGS improves
    fx_usdjpy_change: float = 0.0      # USDJPY % change
    capex: float = 0.0                 # Investment cost
    delta_revenue_pct: float = 0.0     # Revenue change percentage
    opex_change_pct: float = 0.0       # Operating expense % change
    wacc: float = 0.08                 # Discount rate for NPV
    horizon_years: int = 5             # Project life


# Pulls financial data from yfinance
def load_financials_from_yf(ticker: str, years: int = 10) -> pd.DataFrame:

    t = yf.Ticker(ticker)  # Create ticker object

    fin = t.financials.T  # Income statement (transposed for readability)
    bs  = t.balance_sheet.T  # Balance sheet
    cf  = t.cashflow.T  # Cash flow statement

    # Clean up column names: lowercase and replace spaces
    fin.columns = [c.lower().replace(' ', '_') for c in fin.columns]
    bs.columns  = [c.lower().replace(' ', '_') for c in bs.columns]
    cf.columns  = [c.lower().replace(' ', '_') for c in cf.columns]

    # Create DataFrame with rows by year
    df = pd.DataFrame(index=fin.index)

    # Fill important fields (fallbacks used if column name missing)
    df['revenue'] = fin.get('total_revenue', fin.get('revenue'))
    df['cogs'] = -fin.get('cost_of_revenue', fin.get('costofrevenue', 0))  # Negative b/c yfinance stores cost as positive expense
    df['sga'] = -fin.get('selling_general_and_administrative', 0)
    df['rnd'] = -fin.get('research_development', 0)
    df['operating_income'] = fin.get('operating_income', 0)
    df['depreciation'] = -cf.get('depreciation', cf.get('depreciation_and_amortization', 0))
    df['capex'] = -cf.get('capital_expenditure', 0)
    df['working_capital'] = (bs.get('total_current_assets', 0) - bs.get('total_current_liabilities', 0))

    df = df.sort_index().tail(years)  # Take only recent N years
    return df  # Return final dataset


# Load financials from manual CSV instead of yfinance
def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)  # Read CSV file
    df = df.set_index('Year')  # Use "Year" as index

    # Rename to consistent naming
    df = df.rename(columns={
        'Revenue':'revenue','COGS':'cogs','SG&A':'sga','R&D':'rnd',
        'OperatingIncome':'operating_income','Capex':'capex','Depreciation':'depreciation',
        'WorkingCapital':'working_capital'
    })
    return df


# Compute financial KPIs
def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()  # Work on a copy to avoid modifying original

    out['gross_profit'] = out['revenue'] + out['cogs']  # Gross profit
    out['gross_margin'] = out['gross_profit'] / out['revenue']  # Gross margin %

    opex = out['sga'].fillna(0) + out['rnd'].fillna(0)  # Total operating expenses
    out['operating_margin'] = out['operating_income'] / out['revenue']  # Op margin %

    out['cogs_ratio'] = out['cogs'] / out['revenue']  # COGS ratio
    out['sga_ratio'] = out['sga'] / out['revenue']  # SG&A ratio
    out['rnd_ratio'] = out['rnd'] / out['revenue']  # R&D ratio

    tax_rate = 0.25  # Assume 25% tax rate
    ebit = out['operating_income']  # Earnings before interest and taxes

    # Simple capital base estimate = Working capital + accumulated net capex
    capital_base = (
        out['working_capital'].abs()
        + (out['capex'].abs().cumsum() - out['depreciation'].abs().cumsum())
    ).replace(0, np.nan)

    out['roic_proxy'] = (ebit * (1 - tax_rate)) / capital_base  # ROIC proxy formula

    return out  # Return KPI DataFrame

# Apply a scenario to latest financial year
def apply_scenario(latest: Dict[str, float], sc: Scenario) -> Dict[str, float]:
    res = latest.copy()  # Copy dictionary

    cogs_ratio = res['cogs'] / res['revenue']  # Current COGS ratio
    cogs_ratio_new = max(0.0, cogs_ratio - sc.cogs_improvement_pp/100.0)  # Improved COGS ratio
    res['cogs_new'] = cogs_ratio_new * res['revenue']  # New COGS after improvement

    fx_effect_on_op = 0.15 * sc.fx_usdjpy_change/100.0 * res['operating_income']  # Rough FX sensitivity assumption

    revenue_new = res['revenue'] * (1 + sc.delta_revenue_pct/100.0)  # Revenue change %
    opex = res['sga'] + res['rnd']  # Base opex
    opex_new = opex * (1 + sc.opex_change_pct/100.0)  # Adjusted opex

    gross_profit_new = revenue_new - res['cogs_new']  # New gross profit
    op_income_new = gross_profit_new - opex_new + fx_effect_on_op  # New operating income

    # Store all new calculations back into dict
    res.update({
        'revenue_new': revenue_new,
        'opex_new': opex_new,
        'operating_income_new': op_income_new,
        'fx_effect_on_op': fx_effect_on_op
    })
    return res  # Return updated financials


# Compute NPV and IRR of cash flows
def npv_irr(cash_flows, wacc):
    years = np.arange(len(cash_flows))  # [0, 1, 2, ...]
    npv = np.sum([cf / ((1 + wacc)**t) for t, cf in enumerate(cash_flows)])  # Discount each cash flow
    try:
        irr = np.irr(cash_flows)  # Calculate IRR
    except Exception:
        irr = np.nan  # If error, return NaN
    return npv, irr  # Return both NPV & IRR


# Build project free cash flow series
def build_project_cf(latest: Dict[str, float], sc: Scenario) -> pd.Series:
    cfs = [-sc.capex]  # Year 0 = spend capex (negative)
    base_op = latest['operating_income']  # Current operating income
    improved_op = apply_scenario(latest, sc)['operating_income_new']  # New operating income
    delta_op = improved_op - base_op  # Change due to project
    delta_fcf = delta_op * 0.75  # Approx after-tax (25% tax rate)

    for _ in range(sc.horizon_years):  # Add cash flows for each year
        cfs.append(delta_fcf)

    return pd.Series(cfs, name="ProjectCF")  # Convert to Pandas Series


# For charts
def plot_series(series: pd.Series, title: str, out_path: str):
    plt.figure()  
    series.plot(marker="o") 
    plt.title(title)  
    plt.xlabel(series.index.name or "Index")  
    plt.ylabel("Value")  
    plt.tight_layout() 
    plt.savefig(out_path)  
    plt.close() 


# Main execution
def main():
    ap = argparse.ArgumentParser()  # Create argument parser
    ap.add_argument("--ticker", type=str, default="TM")  # Default to Toyota ADR
    ap.add_argument("--years", type=int, default=10)  # Default 10 years
    ap.add_argument("--csv", type=str, default=None, help="Path to manual CSV")  # CSV option
    ap.add_argument("--outdir", type=str, default="outputs_toyota")  # Output directory
    args = ap.parse_args()  # Read args from command line

    os.makedirs(args.outdir, exist_ok=True)  # Create folder if not exists

    if args.csv:  # If CSV provided
        df = load_from_csv(args.csv)
    else:  # Otherwise use yfinance
        df = load_financials_from_yf(args.ticker, args.years)

    kpis = compute_kpis(df)  # Calculate KPIs
    kpis.to_csv(f"{args.outdir}/toyota_kpis.csv")  # Export KPI CSV

    # Create charts for revenue and margins
    plot_series(kpis['revenue'], "Revenue", f"{args.outdir}/revenue.png")
    plot_series(kpis['operating_margin'], "Operating Margin", f"{args.outdir}/operating_margin.png")
    plot_series(kpis['cogs_ratio'], "COGS Ratio", f"{args.outdir}/cogs_ratio.png")

    # Select latest year of data
    latest = kpis.dropna().iloc[-1].to_dict()

    # Example scenario values
    sc = Scenario(
        cogs_improvement_pp=1.0,
        fx_usdjpy_change=10.0,
        capex=500.0,
        delta_revenue_pct=2.0,
        opex_change_pct=0.5,
        wacc=0.08,
        horizon_years=5
    )

    proj_cf = build_project_cf(latest, sc)  # Build cash flows
    proj_cf.to_csv(f"{args.outdir}/project_cf.csv", index=False)  # Save cash flow CSV

    npv, irr = npv_irr(proj_cf.values, sc.wacc)  # Calculate NPV/IRR

    # Write summary file
    with open(f"{args.outdir}/summary.txt", "w", encoding="utf-8") as f:
        f.write("=== Toyota KPI & Project Summary ===\n")
        f.write(f"Years: last {args.years}\n")
        f.write(f"Latest Operating Margin: {latest.get('operating_margin', float('nan')):.2%}\n")
        f.write(f"Scenario: -{sc.cogs_improvement_pp}pp COGS, FX {sc.fx_usdjpy_change}%\n")
        f.write(f"Project Capex: {sc.capex} (JPY bn)\n")
        f.write(f"NPV @ {sc.wacc*100:.1f}%: {npv:.2f}\n")
        f.write(f"IRR: {irr if irr is not None else 'NA'}\n")

# Run main() only if this file is executed directly
if __name__ == "__main__":
    import os  # Imported here so top remains cleaner
    main() 
