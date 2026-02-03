# src/mf/portfolio/metrics.py
import pandas as pd


# ----------------------------
# XIRR (cashflow based)
# ----------------------------
def xirr(cashflows, guess=0.10, max_iter=200, tol=1e-8):
    """
    cashflows: list[(date, amount)]
      BUY = negative, REDEEM/current_value = positive
    """
    if not cashflows or len(cashflows) < 2:
        return None

    cf = []
    for d, a in cashflows:
        dd = pd.to_datetime(d, errors="coerce")
        if pd.isna(dd):
            continue
        try:
            aa = float(a)
        except Exception:
            continue
        cf.append((dd, aa))

    if len(cf) < 2:
        return None

    cf.sort(key=lambda x: x[0])
    d0 = cf[0][0]

    amts = [a for _, a in cf]
    # must contain both negative and positive
    if not (min(amts) < 0 and max(amts) > 0):
        return None

    def npv(rate):
        total = 0.0
        for d, a in cf:
            t = (d - d0).days / 365.0
            total += a / ((1.0 + rate) ** t)
        return total

    def d_npv(rate):
        total = 0.0
        for d, a in cf:
            t = (d - d0).days / 365.0
            if t == 0:
                continue
            total += -t * a / ((1.0 + rate) ** (t + 1.0))
        return total

    r = float(guess)
    for _ in range(max_iter):
        f = npv(r)
        fp = d_npv(r)

        if abs(f) < tol:
            return r
        if fp == 0:
            break

        r_new = r - f / fp
        if r_new <= -0.9999:
            r_new = -0.9999

        if abs(r_new - r) < tol:
            return r_new
        r = r_new

    return None


# ----------------------------
# Helpers
# ----------------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).replace("\ufeff", "").strip().lower().replace(" ", "_")
        for c in df.columns
    ]
    return df


def _ensure_nav_schema(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure nav_df has scheme_code, date, nav.
    """
    nav = _normalize_cols(nav_df)

    if "date" not in nav.columns and "nav_date" in nav.columns:
        nav = nav.rename(columns={"nav_date": "date"})

    need = {"scheme_code", "date", "nav"}
    missing = need - set(nav.columns)
    if missing:
        raise ValueError(f"NAV dataframe missing columns: {missing}. Found: {list(nav.columns)}")

    nav["scheme_code"] = nav["scheme_code"].astype(str).str.strip()
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")

    nav = nav.dropna(subset=["scheme_code", "date", "nav"]).copy()
    nav = nav.sort_values(["scheme_code", "date"]).reset_index(drop=True)
    return nav


def _latest_nav(nav_df: pd.DataFrame) -> pd.DataFrame:
    nav = _ensure_nav_schema(nav_df)
    last = nav.groupby("scheme_code", as_index=False).tail(1)
    last = last.rename(columns={"date": "latest_nav_date", "nav": "latest_nav"})
    return last[["scheme_code", "latest_nav_date", "latest_nav"]]


def _normalize_tx_signs(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Force correct sign based on type:
      BUY/SIP/PURCHASE/INVESTED => negative
      REDEEM/SELL/SWP           => positive
    If type missing -> keep as is.
    """
    tx = tx.copy()
    if "type" not in tx.columns:
        return tx

    t = tx["type"].astype(str).str.strip().str.upper()
    amt = pd.to_numeric(tx["amount"], errors="coerce")

    buy_like = t.isin(["BUY", "SIP", "PURCHASE", "INVESTED"])
    sell_like = t.isin(["REDEEM", "SELL", "SWP"])

    # BUY must be negative
    tx.loc[buy_like, "amount"] = -amt[buy_like].abs()

    # SELL/REDEEM must be positive
    tx.loc[sell_like, "amount"] = amt[sell_like].abs()

    return tx


def _nav_asof_merge_with_fallback(txn: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    """
    Attach NAV to each txn date:
      1) backward asof
      2) forward asof (if still missing)
      3) latest NAV per scheme (final fallback)
    """
    txn = txn.sort_values(["scheme_code", "date"]).copy()
    nav = nav.sort_values(["scheme_code", "date"]).copy()

    out = []
    for code, g in txn.groupby("scheme_code"):
        nav_g = nav[nav["scheme_code"] == code]
        gg = g.sort_values("date").copy()

        if nav_g.empty:
            gg["nav"] = pd.NA
            out.append(gg)
            continue

        # backward
        b = pd.merge_asof(
            gg,
            nav_g[["date", "nav"]].sort_values("date"),
            on="date",
            direction="backward",
        )

        # forward for missing
        missing = b["nav"].isna()
        if missing.any():
            f = pd.merge_asof(
                gg.loc[missing].copy(),
                nav_g[["date", "nav"]].sort_values("date"),
                on="date",
                direction="forward",
            )
            b.loc[missing, "nav"] = f["nav"].values

        out.append(b)

    out_df = pd.concat(out, ignore_index=True) if out else txn.assign(nav=pd.NA)

    # latest fallback
    last = _latest_nav(nav).set_index("scheme_code")["latest_nav"]
    out_df["nav"] = pd.to_numeric(out_df["nav"], errors="coerce")
    out_df.loc[out_df["nav"].isna(), "nav"] = out_df.loc[out_df["nav"].isna(), "scheme_code"].map(last)

    return out_df


def _build_holdings_from_transactions(txn_df: pd.DataFrame, nav_df: pd.DataFrame) -> pd.DataFrame:
    txn = _normalize_cols(txn_df)
    nav = _ensure_nav_schema(nav_df)

    need_txn = {"scheme_code", "date", "amount"}
    missing = need_txn - set(txn.columns)
    if missing:
        raise ValueError(f"Transactions missing columns: {missing}. Found: {list(txn.columns)}")

    if "scheme_name" not in txn.columns:
        txn["scheme_name"] = txn["scheme_code"]

    txn["scheme_code"] = txn["scheme_code"].astype(str).str.strip()
    txn["scheme_name"] = txn["scheme_name"].astype(str).str.strip()
    txn["date"] = pd.to_datetime(txn["date"], errors="coerce")
    txn["amount"] = pd.to_numeric(txn["amount"], errors="coerce")

    txn = txn.dropna(subset=["scheme_code", "scheme_name", "date", "amount"]).reset_index(drop=True)

    # ✅ Fix sign using type (very important)
    txn = _normalize_tx_signs(txn)

    txn_nav = _nav_asof_merge_with_fallback(txn, nav)
    txn_nav["nav"] = pd.to_numeric(txn_nav["nav"], errors="coerce")

    # If type missing, infer
    if "type" not in txn_nav.columns:
        txn_nav["type"] = txn_nav["amount"].apply(lambda x: "BUY" if x < 0 else "REDEEM")

    def units_delta(row):
        navv = row["nav"]
        amt = row["amount"]
        if pd.isna(navv) or navv <= 0:
            return pd.NA

        typ = str(row.get("type", "")).strip().upper()

        # BUY amount is negative => (-amt)/nav positive units
        if typ in ["BUY", "SIP", "PURCHASE", "INVESTED"]:
            return (-amt) / navv

        # REDEEM amount is positive => -(amt)/nav negative units
        if typ in ["REDEEM", "SELL", "SWP"]:
            return -(amt) / navv

        # fallback by sign
        return (-amt) / navv

    txn_nav["units_delta"] = txn_nav.apply(units_delta, axis=1)
    txn_nav["units_delta"] = pd.to_numeric(txn_nav["units_delta"], errors="coerce")
    txn_nav = txn_nav.dropna(subset=["units_delta"]).copy()

    holdings = (
        txn_nav.groupby(["scheme_code", "scheme_name"], as_index=False)["units_delta"]
        .sum()
        .rename(columns={"units_delta": "units"})
    )

    holdings["units"] = pd.to_numeric(holdings["units"], errors="coerce").fillna(0.0)
    holdings = holdings[holdings["units"].abs() > 1e-6].reset_index(drop=True)
    return holdings


def _nav_is_only_latest(nav: pd.DataFrame) -> bool:
    """
    If each scheme_code has only 1 date row, it's basically latest-only NAV.
    """
    g = nav.groupby("scheme_code")["date"].nunique()
    return (g <= 1).all()


# ----------------------------
# Main
# ----------------------------
def compute_portfolio_metrics(txn_df: pd.DataFrame, nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns scheme-wise:
      scheme_code, scheme_name, units, invested, current_value, gain, return_pct, xirr

    txn_df can be:
      A) holdings: scheme_code, scheme_name, units
      B) transactions: scheme_code, date, amount (+ scheme_name optional, type optional)

    nav_df must be:
      scheme_code, date, nav
    """
    df = _normalize_cols(txn_df)
    nav = _ensure_nav_schema(nav_df)
    last_nav = _latest_nav(nav)

    is_holdings = {"scheme_code", "scheme_name", "units"}.issubset(df.columns)
    is_transactions = {"scheme_code", "date", "amount"}.issubset(df.columns)

    # ----------------
    # Holdings path
    # ----------------
    if is_holdings and not is_transactions:
        holdings = df.copy()
        holdings["scheme_code"] = holdings["scheme_code"].astype(str).str.strip()
        holdings["scheme_name"] = holdings["scheme_name"].astype(str).str.strip()
        holdings["units"] = pd.to_numeric(holdings["units"], errors="coerce")
        holdings = holdings.dropna(subset=["scheme_code", "scheme_name", "units"])
        holdings = holdings[holdings["units"] > 0].reset_index(drop=True)

        merged = holdings.merge(last_nav, on="scheme_code", how="left")
        merged["latest_nav"] = pd.to_numeric(merged["latest_nav"], errors="coerce").fillna(0.0)
        merged["current_value"] = merged["units"] * merged["latest_nav"]

        # invested unknown from holdings file
        merged["invested"] = 0.0
        merged["gain"] = merged["current_value"]
        merged["return_pct"] = 0.0
        merged["xirr"] = pd.NA

        return merged[[
            "scheme_code", "scheme_name", "units",
            "invested", "current_value", "gain", "return_pct", "xirr"
        ]].sort_values("current_value", ascending=False).reset_index(drop=True)

    # ----------------
    # Transactions path
    # ----------------
    if is_transactions:
        tx = df.copy()
        tx["scheme_code"] = tx["scheme_code"].astype(str).str.strip()
        if "scheme_name" not in tx.columns:
            tx["scheme_name"] = tx["scheme_code"]
        tx["scheme_name"] = tx["scheme_name"].astype(str).str.strip()
        tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
        tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
        tx = tx.dropna(subset=["scheme_code", "scheme_name", "date", "amount"]).copy()

        # ✅ Fix sign based on type
        tx = _normalize_tx_signs(tx)

        # ⚠️ Detect “latest-only NAV” issue
        if _nav_is_only_latest(nav):
            min_tx_date = tx["date"].min()
            # if txn is before the only nav date, results will be approximate
            # (still compute but user must know)
            # NOTE: do NOT raise exception, just compute.
            # You can change to raise ValueError if you want to hard-stop.

        # base keeps all schemes even if units calc fails
        base = tx[["scheme_code", "scheme_name"]].drop_duplicates().reset_index(drop=True)

        # invested = sum of BUY negative cashflows (absolute)
        invested = (
            tx[tx["amount"] < 0]
            .groupby("scheme_code")["amount"]
            .sum()
            .mul(-1.0)
            .reset_index()
            .rename(columns={"amount": "invested"})
        )

        # holdings (units)
        holdings = _build_holdings_from_transactions(tx, nav)

        merged = base.merge(holdings, on=["scheme_code", "scheme_name"], how="left")
        merged["units"] = pd.to_numeric(merged["units"], errors="coerce").fillna(0.0)

        merged = merged.merge(last_nav, on="scheme_code", how="left")
        merged["latest_nav"] = pd.to_numeric(merged["latest_nav"], errors="coerce").fillna(0.0)
        merged["current_value"] = merged["units"] * merged["latest_nav"]

        merged = merged.merge(invested, on="scheme_code", how="left")
        merged["invested"] = pd.to_numeric(merged["invested"], errors="coerce").fillna(0.0)

        merged["gain"] = merged["current_value"] - merged["invested"]
        merged["return_pct"] = merged.apply(
            lambda r: (r["gain"] / r["invested"] * 100.0) if r["invested"] > 0 else 0.0,
            axis=1
        )

        # Scheme-level XIRR (only meaningful if there are multiple cashflows OR redeem + final value)
        xirr_vals = []
        for _, row in merged.iterrows():
            scode = row["scheme_code"]
            scheme_tx = tx[tx["scheme_code"] == scode].sort_values("date")
            cfs = list(zip(scheme_tx["date"].tolist(), scheme_tx["amount"].tolist()))

            as_of = row.get("latest_nav_date", None)
            cv = float(row.get("current_value", 0.0))
            if pd.notna(as_of) and cv > 0:
                cfs.append((as_of, cv))

            xirr_vals.append(xirr(cfs))

        merged["xirr"] = xirr_vals

        return merged[[
            "scheme_code", "scheme_name", "units",
            "invested", "current_value", "gain", "return_pct", "xirr"
        ]].sort_values("current_value", ascending=False).reset_index(drop=True)

    raise ValueError(
        "CSV not recognized.\n"
        "Provide either holdings columns: scheme_code, scheme_name, units\n"
        "OR transactions columns: scheme_code, date, amount"
    )
