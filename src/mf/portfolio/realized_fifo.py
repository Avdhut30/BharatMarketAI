# src/mf/portfolio/realized_fifo.py
import pandas as pd

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff","").strip().lower().replace(" ","_") for c in df.columns]
    return df

def compute_realized_unrealized_fifo(
    tx_df: pd.DataFrame,
    nav_latest_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    FIFO realized/unrealized per scheme.
    Expects tx_df columns: scheme_code, scheme_name(optional), date, amount, units(optional), type(optional)
      - BUY: amount negative, units positive (if units missing we can't compute FIFO)
      - REDEEM: amount positive, units negative (or redeem units specified as positive, handled)
    If units not present in tx_df, this function cannot compute realized reliably.
    """

    tx = _norm_cols(tx_df)
    nav = _norm_cols(nav_latest_df)

    if "scheme_name" not in tx.columns:
        tx["scheme_name"] = tx["scheme_code"]

    req = {"scheme_code","date","amount"}
    if not req.issubset(tx.columns):
        raise ValueError(f"Need transactions with {req}. Found: {list(tx.columns)}")

    # units required for realized
    if "units" not in tx.columns:
        # return empty with message-friendly schema
        return pd.DataFrame(columns=[
            "scheme_code","scheme_name","realized_gain","unrealized_gain","cost_remaining","units_remaining"
        ])

    tx["scheme_code"] = tx["scheme_code"].astype(str).str.strip()
    tx["scheme_name"] = tx["scheme_name"].astype(str).str.strip()
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
    tx["units"] = pd.to_numeric(tx["units"], errors="coerce")

    tx = tx.dropna(subset=["scheme_code","date","amount","units"]).sort_values(["scheme_code","date"])

    # latest nav map
    if "latest_nav" in nav.columns:
        nav_map = nav.set_index("scheme_code")["latest_nav"].to_dict()
    elif "nav" in nav.columns:
        nav_map = nav.set_index("scheme_code")["nav"].to_dict()
    else:
        raise ValueError("nav_latest_df must contain either latest_nav or nav")

    rows = []

    for scode, g in tx.groupby("scheme_code"):
        sname = g["scheme_name"].iloc[0]
        lots = []  # each lot: [units_remaining, cost_per_unit]
        realized = 0.0

        for _, r in g.iterrows():
            amt = float(r["amount"])
            u = float(r["units"])

            # Normalize: BUY should increase units; REDEEM should reduce units
            # If user stores REDEEM units as positive, we convert to negative using amount sign.
            if amt < 0 and u < 0:
                u = abs(u)
            if amt > 0 and u > 0:
                u = -abs(u)

            if amt < 0:  # BUY
                buy_units = u
                if buy_units <= 0:
                    continue
                cost_per_unit = (-amt) / buy_units
                lots.append([buy_units, cost_per_unit])

            else:  # REDEEM
                sell_units = -u
                if sell_units <= 0:
                    continue
                proceeds = amt
                # estimate sell price per unit using proceeds / units
                sell_price = proceeds / sell_units

                remaining_to_sell = sell_units
                while remaining_to_sell > 1e-9 and lots:
                    lot_units, lot_cpu = lots[0]
                    take = min(lot_units, remaining_to_sell)
                    realized += take * (sell_price - lot_cpu)
                    lot_units -= take
                    remaining_to_sell -= take
                    if lot_units <= 1e-9:
                        lots.pop(0)
                    else:
                        lots[0][0] = lot_units

        units_remaining = sum(u for u, _ in lots)
        cost_remaining = sum(u * cpu for u, cpu in lots)
        latest_nav = float(nav_map.get(scode, 0.0))
        current_value = units_remaining * latest_nav
        unrealized = current_value - cost_remaining

        rows.append({
            "scheme_code": scode,
            "scheme_name": sname,
            "realized_gain": realized,
            "unrealized_gain": unrealized,
            "cost_remaining": cost_remaining,
            "units_remaining": units_remaining,
        })

    return pd.DataFrame(rows)
