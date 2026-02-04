# src/mf/ui/mf_pages.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.mf.portfolio.pdf_export import build_mf_summary_pdf
from src.mf.portfolio.realized_fifo import compute_realized_unrealized_fifo

from src.mf.portfolio.metrics import compute_portfolio_metrics, xirr
from src.mf.portfolio.profile_store import (
    load_profile, save_profile, get_scheme_tags,
    ASSET_CLASSES, DEFAULT_GOALS
)
from src.mf.portfolio.rebalancing import allocation_table, compute_drift, suggestion_add_next
from src.mf.portfolio.sip import detect_sips, sip_calendar, sip_health


DATA_DIR = "data_cache"
DEFAULT_TXN_PATH = os.path.join(DATA_DIR, "mf_transaction.csv")
DEFAULT_HOLDINGS_PATH = os.path.join(DATA_DIR, "mf_holdings.csv")


def _currency(x: float) -> str:
    try:
        return f"₹{float(x):,.0f}"
    except Exception:
        return "₹0"


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _load_default_csv_if_exists():
    """
    Loads local CSV automatically if user does not upload.
    Priority:
      1) mf_transaction.csv
      2) mf_holdings.csv
    """
    if os.path.exists(DEFAULT_TXN_PATH):
        df = pd.read_csv(DEFAULT_TXN_PATH)
        return df, DEFAULT_TXN_PATH
    if os.path.exists(DEFAULT_HOLDINGS_PATH):
        df = pd.read_csv(DEFAULT_HOLDINGS_PATH)
        return df, DEFAULT_HOLDINGS_PATH
    return None, None


def mf_dashboard():
    st.header("📈 Mutual Fund Dashboard (MF Central Style)")
    st.caption(
        "CSV-first portfolio view: upload transactions/holdings → NAV history → "
        "portfolio value, XIRR, goals, rebalancing, SIP tracking."
    )

    st.markdown("### ✅ Data Source")
    uploaded = st.file_uploader("Upload MF CSV (transactions or holdings)", type=["csv"])

    default_df, default_path = (None, None)
    if not uploaded:
        default_df, default_path = _load_default_csv_if_exists()
        if default_df is not None:
            st.info(f"Using local file automatically: **{default_path}**")
        else:
            st.warning(
                "No file uploaded and no local CSV found.\n\n"
                "Create one of these:\n"
                f"- `{DEFAULT_TXN_PATH}` (transactions)\n"
                f"- `{DEFAULT_HOLDINGS_PATH}` (holdings)\n"
            )
            return

    with st.expander("📄 Supported CSV formats", expanded=False):
        st.markdown("**A) Transactions CSV (recommended)**")
        st.code(
            "date,scheme_code,scheme_name,amount,type\n"
            "2022-01-10,INF179K01WZ0,Parag Parikh Flexi Cap Fund,-10000,BUY\n"
            "2023-02-10,INF179K01WZ0,Parag Parikh Flexi Cap Fund,12000,REDEEM",
            language="text",
        )
        st.markdown("**B) Holdings CSV**")
        st.code(
            "scheme_code,scheme_name,units\n"
            "INF179K01WZ0,Parag Parikh Flexi Cap Fund,158.25\n"
            "INF090I01SE0,HDFC Balanced Advantage Fund,95.10",
            language="text",
        )

    # ----------------------------
    # Load CSV (transactions or holdings)
    # ----------------------------
    from src.mf.portfolio.importer_csv import load_mf_csv

    if uploaded:
        txn_df = load_mf_csv(uploaded)
    else:
        txn_df = default_df.copy()

    txn_df = _normalize_cols(txn_df)

    # ----------------------------
    # Decide: holdings vs transactions
    # ----------------------------
    is_holdings = {"scheme_code", "scheme_name", "units"}.issubset(txn_df.columns)
    is_txn = {"scheme_code", "date", "amount"}.issubset(txn_df.columns)

    # ----------------------------
    # Extract scheme codes (ISINs) from CSV
    # ----------------------------
    isins = []
    if "scheme_code" in txn_df.columns:
        isins = (
            txn_df["scheme_code"]
            .astype(str)
            .str.strip()
            .replace("nan", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )

    if not isins:
        st.error("No scheme_code found in your CSV.")
        st.stop()

    st.divider()

    # ----------------------------
    # ✅ NAV HISTORY fetch + refresh control
    # ----------------------------
    st.subheader("📡 NAV Sync")
    col1, col2 = st.columns([1, 2])
    force_refresh = col1.button("🔄 Refresh NAV now")
    col2.caption("Uses cached NAV by default. Click refresh if you want latest NAV immediately.")

    from src.mf.data.nav_history import get_nav_history_for_isins
    nav_df, meta = get_nav_history_for_isins(isins, force_refresh=force_refresh, max_years=10)

    if nav_df.empty:
        st.error(
            "Could not fetch NAV history for your funds.\n\n"
            "Possible reasons:\n"
            "1) scheme_code is not ISIN (INFxxxx)\n"
            "2) internet blocked\n"
            "3) AMFI / mfapi temporarily down\n"
            "4) Too many requests (rate limiting)"
        )
        st.stop()

    if meta.get("missing_isins"):
        st.warning(
            "Some scheme_code values were NOT found in AMFI master, so they were skipped:\n"
            + ", ".join(meta["missing_isins"])
        )

    nav_df = nav_df[["scheme_code", "scheme_name", "date", "nav"]].copy()

    # ----------------------------
    # ✅ Compute summary using NAV HISTORY
    # ----------------------------
    summary = compute_portfolio_metrics(txn_df, nav_df)

    # ----------------------------
    # Load profile tags
    # ----------------------------
    profile = load_profile()

    assets, goals = [], []
    for _, r in summary.iterrows():
        asset, goal = get_scheme_tags(profile, str(r["scheme_code"]), r["scheme_name"])
        assets.append(asset)
        goals.append(goal)

    summary["asset_class"] = assets
    summary["goal"] = goals

    # ----------------------------
    # Portfolio KPIs
    # ----------------------------
    total_invested = float(pd.to_numeric(summary["invested"], errors="coerce").fillna(0.0).sum())
    total_current = float(pd.to_numeric(summary["current_value"], errors="coerce").fillna(0.0).sum())
    total_gain = total_current - total_invested
    total_return_pct = (total_gain / total_invested * 100) if total_invested > 0 else 0.0

    # ----------------------------
    # ✅ Portfolio XIRR (transactions only)
    # ----------------------------
    port_xirr = None
    if is_txn:
        tx = txn_df.copy()
        tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
        tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
        tx = tx.dropna(subset=["date", "amount"]).sort_values("date")
        cfs = list(zip(tx["date"], tx["amount"]))  # BUY negative, REDEEM positive
        cfs.append((pd.Timestamp.today(), total_current))
        port_xirr = xirr(cfs)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Invested", _currency(total_invested))
    c2.metric("Current Value", _currency(total_current))
    c3.metric("Gain / Loss", _currency(total_gain), f"{total_return_pct:.2f}%")
    c4.metric("Portfolio XIRR", f"{(port_xirr * 100):.2f}%" if port_xirr is not None else "N/A")

    st.divider()

    # ----------------------------
    # ✅ CAGR per scheme (only for transactions)
    # ----------------------------
    summary["cagr"] = pd.NA
    if is_txn:
        tx = txn_df.copy()
        tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
        tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
        tx = tx.dropna(subset=["scheme_code", "date", "amount"]).copy()
        tx["scheme_code"] = tx["scheme_code"].astype(str).str.strip()

        first_dt = tx.groupby("scheme_code")["date"].min().reset_index().rename(columns={"date": "first_invest_date"})
        tmp = summary.merge(first_dt, on="scheme_code", how="left")

        years = (pd.Timestamp.today().normalize() - pd.to_datetime(tmp["first_invest_date"], errors="coerce")).dt.days / 365.0
        years = years.clip(lower=0.1)

        inv = pd.to_numeric(tmp["invested"], errors="coerce").fillna(0.0)
        cur = pd.to_numeric(tmp["current_value"], errors="coerce").fillna(0.0)

        cagr = (cur / inv) ** (1 / years) - 1
        cagr[(inv <= 0) | (cur <= 0) | (years <= 0)] = pd.NA

        summary["cagr"] = cagr

    # ----------------------------
    # ✅ Filters (search / hide zero / asset / goal)
    # ----------------------------
    st.subheader("🔎 Filters")
    f1, f2, f3, f4 = st.columns([2, 1.5, 1.5, 1.2])

    q = f1.text_input("Search scheme name", value="")
    hide_zero = f2.checkbox("Hide zero units", value=True)

    asset_options = sorted([x for x in summary["asset_class"].dropna().unique().tolist() if str(x).strip()])
    goal_options = sorted([x for x in summary["goal"].dropna().unique().tolist() if str(x).strip()])

    asset_sel = f3.multiselect("Asset class", asset_options, default=[])
    goal_sel = f4.multiselect("Goal", goal_options, default=[])

    filtered = summary.copy()

    if q.strip():
        filtered = filtered[filtered["scheme_name"].astype(str).str.contains(q.strip(), case=False, na=False)]

    if hide_zero and "units" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["units"], errors="coerce").fillna(0.0) != 0.0]

    if asset_sel:
        filtered = filtered[filtered["asset_class"].isin(asset_sel)]

    if goal_sel:
        filtered = filtered[filtered["goal"].isin(goal_sel)]

    st.divider()

    # ----------------------------
    # Classify funds
    # ----------------------------
    st.subheader("🏷️ Classify your funds (Asset Class + Goal)")
    st.caption("Saved locally in data_cache/mf_profile.json")

    goal_list = DEFAULT_GOALS.copy()
    custom_goals = st.text_input("Add custom goals (comma-separated)", value="")
    if custom_goals.strip():
        goal_list += [g.strip() for g in custom_goals.split(",") if g.strip()]
        goal_list = list(dict.fromkeys(goal_list))

    tag_df = summary[["scheme_code", "scheme_name", "asset_class", "goal"]].copy()
    tag_df["scheme_code"] = tag_df["scheme_code"].astype(str)

    edited = st.data_editor(
        tag_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "asset_class": st.column_config.SelectboxColumn("asset_class", options=ASSET_CLASSES),
            "goal": st.column_config.SelectboxColumn("goal", options=goal_list),
        },
        hide_index=True,
    )

    if st.button("💾 Save classification"):
        for _, row in edited.iterrows():
            scode = str(row["scheme_code"])
            profile[scode] = {
                "asset_class": row["asset_class"],
                "goal": row["goal"],
                "scheme_name": row["scheme_name"],
            }
        save_profile(profile)
        st.success("Saved classification ✅")

    summary = summary.merge(
        edited[["scheme_code", "asset_class", "goal"]],
        on="scheme_code",
        how="left",
        suffixes=("", "_new"),
    )
    summary["asset_class"] = summary["asset_class_new"].fillna(summary["asset_class"])
    summary["goal"] = summary["goal_new"].fillna(summary["goal"])
    summary = summary.drop(columns=["asset_class_new", "goal_new"])

    # Re-apply filters after edits
    filtered = summary.copy()
    if q.strip():
        filtered = filtered[filtered["scheme_name"].astype(str).str.contains(q.strip(), case=False, na=False)]
    if hide_zero and "units" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["units"], errors="coerce").fillna(0.0) != 0.0]
    if asset_sel:
        filtered = filtered[filtered["asset_class"].isin(asset_sel)]
    if goal_sel:
        filtered = filtered[filtered["goal"].isin(goal_sel)]

    st.divider()

    # ----------------------------
    # Scheme table
    # ----------------------------
    st.subheader("📊 Scheme-wise Summary")

    show_cols = [
        "scheme_name",
        "asset_class",
        "goal",
        "units",
        "invested",
        "current_value",
        "gain",
        "return_pct",
        "cagr",
        "xirr",
    ]

    out = filtered.copy()
    for c in ["units", "invested", "current_value", "gain", "return_pct", "cagr", "xirr"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    st.dataframe(
        out[show_cols].sort_values("current_value", ascending=False),
        use_container_width=True,
        height=650,
    )

    st.download_button(
        "⬇️ Download scheme summary",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="mf_scheme_summary.csv",
        mime="text/csv",
    )

    st.divider()

    # ----------------------------
    # ✅ Allocation (PIE charts + tables)
    # ----------------------------
    st.subheader("🍰 Allocation")

    left, right = st.columns(2)

    with left:
        st.write("Allocation by Asset Class")
        alloc_asset = allocation_table(summary, "asset_class")
        st.dataframe(alloc_asset, use_container_width=True, height=350)

        if not alloc_asset.empty and "weight" in alloc_asset.columns:
            fig, ax = plt.subplots()
            ax.pie(
                alloc_asset["weight"],
                labels=alloc_asset["asset_class"].astype(str),
                autopct="%1.1f%%",
            )
            ax.set_title("Asset Class %")
            st.pyplot(fig)

    with right:
        st.write("Allocation by Goal")
        alloc_goal = allocation_table(summary, "goal")
        st.dataframe(alloc_goal, use_container_width=True, height=350)

        if not alloc_goal.empty and "weight" in alloc_goal.columns:
            fig, ax = plt.subplots()
            ax.pie(
                alloc_goal["weight"],
                labels=alloc_goal["goal"].astype(str),
                autopct="%1.1f%%",
            )
            ax.set_title("Goal %")
            st.pyplot(fig)

    st.divider()

    # ----------------------------
    # Rebalancing
    # ----------------------------
    st.subheader("⚖️ Rebalancing (Asset Allocation Drift)")
    st.caption("Set a target allocation and check drift. Research-only portfolio tool.")

    colA, colB, colC, colD, colE = st.columns(5)
    t_equity = colA.number_input("Equity %", min_value=0, max_value=100, value=60, step=5)
    t_debt = colB.number_input("Debt %", min_value=0, max_value=100, value=30, step=5)
    t_hybrid = colC.number_input("Hybrid %", min_value=0, max_value=100, value=10, step=5)
    t_gold = colD.number_input("Gold %", min_value=0, max_value=100, value=0, step=5)
    t_intl = colE.number_input("Intl %", min_value=0, max_value=100, value=0, step=5)

    target_total = t_equity + t_debt + t_hybrid + t_gold + t_intl
    if target_total != 100:
        st.warning("Target allocation must total 100%. Adjust the inputs.")
    else:
        target = {
            "Equity": t_equity / 100.0,
            "Debt": t_debt / 100.0,
            "Hybrid": t_hybrid / 100.0,
            "Gold": t_gold / 100.0,
            "International": t_intl / 100.0,
        }

        alloc_asset = allocation_table(summary, "asset_class")
        drift = compute_drift(alloc_asset, target, col_name="asset_class")
        st.dataframe(drift, use_container_width=True, height=400)

        add_next = suggestion_add_next(drift)
        if add_next:
            st.success(f"Suggestion ✅ Add more to: **{add_next}** (below target)")
        else:
            st.info("Close to target allocation.")

    st.divider()

    # ----------------------------
    # ✅ Realized vs Unrealized (FIFO)
    # ----------------------------
    with st.expander("💰 Realized vs Unrealized (FIFO)"):
        if not is_txn:
            st.info("This section needs Transactions CSV.")
        else:
            if "units" not in txn_df.columns:
                st.warning("Your transactions CSV does not include `units` column. FIFO needs units.")
            else:
                try:
                    ru = compute_realized_unrealized_fifo(txn_df, nav_df)
                    if isinstance(ru, pd.DataFrame) and not ru.empty:
                        st.dataframe(ru, use_container_width=True, height=450)
                    else:
                        st.info("No realized/unrealized rows computed.")
                except Exception as e:
                    st.error(f"FIFO calculation failed: {e}")

    st.divider()

    # ----------------------------
    # SIP Tracker
    # ----------------------------
    st.subheader("📅 SIP Tracker")

    if is_txn:
        sip_df = detect_sips(txn_df)
        if sip_df.empty:
            st.info("No SIPs detected.")
        else:
            sip_df = sip_health(sip_df)
            st.dataframe(sip_df, use_container_width=True, height=350)

            st.subheader("🗓️ SIP Calendar")
            cal = sip_calendar(txn_df)
            st.dataframe(cal, use_container_width=True, height=350)

            cal2 = cal.copy()
            cal2["ym"] = cal2["year"].astype(str) + "-" + cal2["month"].astype(str).str.zfill(2)
            st.bar_chart(cal2.set_index("ym")["sip_amount"])
    else:
        st.info("SIP tracker needs transactions CSV with date, amount, scheme_code.")

    st.divider()

    # ----------------------------
    # ✅ Export PDF Summary
    # ----------------------------
    st.subheader("📄 Export Portfolio PDF")

    pdf_rows = [("Scheme", "Units", "Invested", "Current", "Gain", "CAGR")]
    tmp = summary.copy()
    for c in ["units", "invested", "current_value", "gain", "cagr"]:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp = tmp.sort_values("current_value", ascending=False).head(25)

    for _, r in tmp.iterrows():
        cagr_txt = ""
        if pd.notna(r.get("cagr")):
            cagr_txt = f"{float(r['cagr'])*100:.2f}%"
        pdf_rows.append((
            str(r.get("scheme_name", ""))[:55],
            f"{float(r.get('units', 0.0)):.2f}",
            _currency(r.get("invested", 0.0)),
            _currency(r.get("current_value", 0.0)),
            _currency(r.get("gain", 0.0)),
            cagr_txt,
        ))

    try:
        pdf_bytes = build_mf_summary_pdf(
            title="Mutual Fund Portfolio Summary",
            kpis={
                "Invested": _currency(total_invested),
                "Current Value": _currency(total_current),
                "Gain": _currency(total_gain),
                "Return %": f"{total_return_pct:.2f}%",
                "XIRR": f"{(port_xirr*100):.2f}%" if port_xirr is not None else "N/A",
            },
            table_rows=pdf_rows,
        )

        st.download_button(
            "⬇️ Download PDF Summary",
            data=pdf_bytes,
            file_name="mf_portfolio_summary.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.error(f"PDF export failed: {e}")
