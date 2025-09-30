
# app.py
# Streamlit Adaptive Pricing Experiment (simple, production-ready prototype)
import time
import os
import uuid
import math
from datetime import datetime

import pandas as pd
import streamlit as st

# ---------- CONFIG (edit these if you want) ----------
APP_TITLE = "Adaptive Pricing Experiment"
NUM_PRODUCTS_TO_SHOW = 5          # how many products per session
UP_PCT = 0.12                     # +12% if ( would buy)
DOWN_PCT = 0.08                   # -8% otherwise (wouldn't buy OR skip)
MIN_MULTIPLIER = 0.50             # don't go below 50% of base
MAX_MULTIPLIER = 2.00    
# --- Time-aware step sizes (tune as you like) ---
FAST_DECISION_MS = 2000   # < 2s = "fast"
UP_FAST   = 0.15          # fast YES -> +15%
UP_SLOW   = 0.08          # slow  YES -> +8%
DOWN_FAST = 0.15          # fast NO  -> -15%
DOWN_SLOW = 0.08          # slow  NO  -> -8%
HESITATION_MIN_FLIPS = 2  # treat as "slow" if user flips answers >= 2 times
         # don't go above 200% of base

DATA_DIR = "data"
TX_CSV = os.path.join(DATA_DIR, "transactions.csv")
SURVEY_CSV = os.path.join(DATA_DIR, "surveys.csv")
PRODUCTS_CSV = "products.csv"

# ----------------------------------------------------

# Ensure data dir
os.makedirs(DATA_DIR, exist_ok=True)

def load_products():
    """Load products from CSV or fallback to defaults."""
    if os.path.exists(PRODUCTS_CSV):
        df = pd.read_csv(PRODUCTS_CSV)

        # Basic validation
        if "name" not in df.columns or "base_price" not in df.columns:
            st.error("products.csv must include at least 'name' and 'base_price' columns.")
            st.stop()

        # Fill / normalize columns
        if "id" not in df.columns:
            df["id"] = [f"p{i+1}" for i in range(len(df))]
        if "image_url" not in df.columns:
            df["image_url"] = ""

        df["id"] = df["id"].astype(str).str.strip()
        df["name"] = df["name"].astype(str).str.strip()
        df["image_url"] = df["image_url"].astype(str).fillna("").str.strip()
        df["base_price"] = pd.to_numeric(df["base_price"], errors="coerce")

        df = df.dropna(subset=["base_price"]).reset_index(drop=True)
        return df[["id", "name", "base_price", "image_url"]].to_dict("records")

    # Fallback default list (used only if products.csv is missing)
    return [
        {"id": "p1", "name": "Wireless Headphones", "base_price": 79.0,
         "image_url": "images/headphones.jpg"},
        {"id": "p2", "name": "Electric Kettle", "base_price": 39.0,
         "image_url": "images/kettle.jpg"},
        {"id": "p3", "name": "Backpack", "base_price": 49.0,
         "image_url": "images/backpack.jpg"},
        {"id": "p4", "name": "Bluetooth Speaker", "base_price": 59.0,
         "image_url": "images/speaker.jpg"},
        {"id": "p5", "name": "Desk Lamp", "base_price": 29.0,
         "image_url": "images/lamp.jpg"},
    ]
    

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def round_price_eur(x):
    """Round to 2 decimals for EUR."""
    return float(f"{x:.2f}")

def ensure_csv(path, columns):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)

def append_rows(path, rows, columns):
    ensure_csv(path, columns)
    df = pd.DataFrame(rows, columns=columns)
    # mode='a' appends; header=False to avoid repeating header
    df.to_csv(path, mode="a", header=False, index=False)

def read_tx():
    cols = [
        "timestamp","session_id","product_id","product_name","base_price",
        "offered_price","fair","would_buy","bought","revenue","lost_revenue",
        "decision_ms","fair_changes","buy_changes","up_pct","down_pct"
    ]
    if not os.path.exists(TX_CSV):
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(TX_CSV)
    except pd.errors.ParserError:
        # Fallback for malformed rows: skip bad lines, then continue normalization
        df = pd.read_csv(TX_CSV, engine="python", on_bad_lines="skip")

    # --- Sanitize / normalize types ---
    for c in ["product_id", "session_id", "product_name"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # ensure booleans are real booleans (CSV may store as "True"/"False" strings)
    if "bought" in df.columns and df["bought"].dtype != bool:
        df["bought"] = df["bought"].astype(str).str.lower().isin(["true", "1", "yes"])

    # numeric columns
    for c in ["offered_price", "base_price", "revenue", "lost_revenue", "decision_ms",
              "up_pct", "down_pct", "fair_changes", "buy_changes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ensure expected columns exist even if skipped lines dropped some
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    # keep only known columns to avoid downstream mismatches
    df = df[cols]

    return df

def others_avg_paid_by_product(exclude_session_id):
    df = read_tx()
    if df.empty:
        return {}
    # Exclude current session
    df_excl = df[df["session_id"] != exclude_session_id]
    if df_excl.empty:
        return {}

    # Primary: average among purchases only
    bought_mask = df_excl["bought"] == True
    df_bought = df_excl[bought_mask]
    bought_means = df_bought.groupby("product_id")["offered_price"].mean().to_dict() if not df_bought.empty else {}

    # Fallback: if a product has no purchases, compute average of all shown prices
    all_means = df_excl.groupby("product_id")["offered_price"].mean().to_dict()

    # Merge: prefer bought_means, else fallback to all_means
    result = {pid: bought_means.get(pid, all_means.get(pid, None)) for pid in all_means.keys()}
    return result

# ---------- SESSION STATE ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "products" not in st.session_state:
    all_products = load_products()
    # Truncate or use all
    st.session_state.products = all_products[:NUM_PRODUCTS_TO_SHOW]

if "idx" not in st.session_state:
    st.session_state.idx = 0  # which product we are on (0-based)

if "multiplier" not in st.session_state:
    st.session_state.multiplier = 1.0  # global price factor applied to next product

if "history" not in st.session_state:
    st.session_state.history = []  # list of dict rows per product decision

if "finished" not in st.session_state:
    st.session_state.finished = False  # whether we reached the survey page

if "start_ts" not in st.session_state:
    st.session_state.start_ts = 0.0

if "fair_changes" not in st.session_state:
    st.session_state.fair_changes = 0

if "buy_changes" not in st.session_state:
    st.session_state.buy_changes = 0

if "touched_fair" not in st.session_state:
    st.session_state.touched_fair = False

if "touched_buy" not in st.session_state:
    st.session_state.touched_buy = False


# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, page_icon="💶", layout="centered")
st.title(APP_TITLE)
st.caption("Show a product → ask fairness & purchase → adapt prices in real time to maximize revenue.")
# --- Styling helpers for bigger questions ---
st.markdown("""
<style>
.big-q { font-size: 1.6rem; font-weight: 400; margin: 0.5rem 0 0.25rem; }
.big-q.secondary { font-size: 1.3rem; font-weight: 500; margin: 0.75rem 0 0.25rem; }

/* Make Yes/No options a bit larger too */
.stRadio [role="radiogroup"] > label { font-size: 1.6rem; }

/* Slightly larger helper text under inputs (optional) */
.block-container .stTooltipContent, .stCaption, .stMarkdown p { font-size: 1.0rem; }
</style>
""", unsafe_allow_html=True)


SHOW_DEBUG = False  # change to False before real experiment

# UI tweaks: enlarge question and buttons
st.markdown(
    """
    <style>
    /* Enlarge radio (main question) labels */
    div.stRadio > label { font-size: 2.10rem !important; }
    div[role="radiogroup"] label { font-size: 2.10rem !important; }
    /* Enlarge questionnaire slider labels */
    div.stSlider > label { font-size: 2.00rem !important; }
    div.stSlider label { font-size: 2.00rem !important; }
    /* Enlarge headings */
    h2, h3 { font-size: 2.2rem !important; }
    /* Enlarge dataframe font */
    div[data-testid="stDataFrame"] * { font-size: 1.20rem !important; }
    /* Enlarge buttons */
    .stButton > button { font-size: 1.2rem; padding: 0.80rem 1.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if SHOW_DEBUG:
    with st.expander("⚙️ Admin / Debug"):
        colA, colB, colC = st.columns(3)
        with colA:
            st.write(f"Session: `{st.session_state.session_id}`")
            st.write(f"Current multiplier: {st.session_state.multiplier:.3f}")
        with colB:
            if st.button("🔁 Reset session"):
                for k in ["idx","multiplier","history","finished"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        with colC:
            st.write("UP %:", UP_PCT)
            st.write("DOWN %:", DOWN_PCT)


# If finished, show survey page
if st.session_state.finished or st.session_state.idx >= len(st.session_state.products):
    st.session_state.finished = True
    st.header("Final Questions & Summary")

    # Build your history DataFrame
    hist_df = pd.DataFrame(st.session_state.history)
    if hist_df.empty:
        st.info("No interactions recorded in this session.")
    else:
        # Compute “others paid” averages
        others = others_avg_paid_by_product(st.session_state.session_id)
        hist_df["others_avg_paid"] = hist_df["product_id"].map(others).fillna(pd.NA)

        # Show what you paid vs others
        display_cols = ["product_name","base_price","offered_price","bought","others_avg_paid"]
        st.subheader("What you saw vs. what others paid (avg)")
        st.dataframe(hist_df[display_cols].rename(columns={
            "product_name":"Product",
            "base_price":"Base €",
            "offered_price":"You saw €",
            "bought":"You bought?",
            "others_avg_paid":"Others avg paid €"
        }), use_container_width=True)

        # Totals
        total_revenue = float(hist_df["revenue"].sum())
        total_lost = float(hist_df["lost_revenue"].sum())
        acceptance_rate = (hist_df["bought"].mean() * 100.0) if len(hist_df) else 0.0

        if SHOW_DEBUG:
            c1, c2, c3 = st.columns(3)
            c1.metric("Your revenue (this session)", f"€{total_revenue:.2f}")
            c2.metric("Lost revenue (skips/unfair/no-buy)", f"€{total_lost:.2f}")
            c3.metric("Acceptance rate", f"{acceptance_rate:.1f}%")

        st.divider()
        st.subheader("Perception of dynamic pricing")

        st.markdown("<div class='big-q secondary'>How fair do you feel the pricing process was overall?</div>", unsafe_allow_html=True)
        fairness_score = st.slider(
            "",
            min_value=0, max_value=10, value=5, help="0 = Not fair at all, 10 = Very fair",
            label_visibility="collapsed",
        )
        st.markdown("<div class='big-q secondary'>How comfortable are you with prices changing dynamically (by time/user)?</div>", unsafe_allow_html=True)
        satisfaction_score = st.slider(
            "",
            min_value=0, max_value=10, value=5, help="0 = Not comfortable, 10 = Very comfortable",
            label_visibility="collapsed",
        )
        st.markdown("<div class='big-q secondary'>How transparent did the pricing feel?</div>", unsafe_allow_html=True)
        price_sensitivity = st.slider(
            "",
            min_value=0, max_value=10, value=5, help="0 = Not transparent, 10 = Very transparent",
            label_visibility="collapsed",
        )
        st.markdown("<div class='big-q secondary'>Is it fair for software to adjust your price (sometimes higher, sometimes discounted) based on predicted willingness to pay?</div>", unsafe_allow_html=True)
        personalized_fairness = st.slider(
            "",
            min_value=0, max_value=10, value=5, help="0 = Not fair at all, 10 = Completely fair",
            label_visibility="collapsed",
        )
        comments = st.text_area("Any thoughts on dynamic pricing, fairness, or trust?")

        if st.button("Submit survey and save"):
            # Save transactions (one row per product)
            tx_columns = ["timestamp","session_id","product_id","product_name","base_price",
              "offered_price","fair","would_buy","bought","revenue","lost_revenue",
              "decision_ms","fair_changes","buy_changes","up_pct","down_pct"]

            tx_rows = []
            now = datetime.utcnow().isoformat()
            for row in st.session_state.history:
                tx_rows.append([
                now,
                st.session_state.session_id,
                row["product_id"],
                row["product_name"],
                row["base_price"],
                row["offered_price"],
                row["fair"],
                row["would_buy"],
                row["bought"],
                row["revenue"],
                row["lost_revenue"],
                row.get("decision_ms", None),
                row.get("fair_changes", None),
                row.get("buy_changes", None),
                UP_PCT,
                DOWN_PCT
            ])

            append_rows(TX_CSV, tx_rows, tx_columns)

            # Save survey (one row per session)
            survey_columns = ["timestamp","session_id","total_revenue","total_lost",
                              "fairness_score","satisfaction_score","price_sensitivity","comments"]
            enriched_comments = comments
            try:
                enriched_comments = f"{comments}\n[personalized_pricing_fairness={personalized_fairness}]"
            except Exception:
                pass
            survey_row = [[now, st.session_state.session_id, total_revenue, total_lost,
                           fairness_score, satisfaction_score, price_sensitivity, enriched_comments]]
            append_rows(SURVEY_CSV, survey_row, survey_columns)

            st.success("Thanks! Your responses were saved. You can close this tab.")
            st.stop()

    st.stop()

# ---------- MAIN FLOW: show current product ----------
products = st.session_state.products
idx = st.session_state.idx
product = products[idx]

# Start timing + reset hesitation counters for this product
st.session_state.start_ts = time.time()
st.session_state.fair_changes = 0
st.session_state.buy_changes = 0
st.session_state.touched_fair = False
st.session_state.touched_buy = False

base = float(product["base_price"])
offered = round_price_eur(base * st.session_state.multiplier)
discount_pct = 0.0
if offered < base:
    discount_pct = (1 - offered / base) * 100.0

# UI
st.progress((idx) / len(products))
st.subheader(f"Product {idx+1} of {len(products)}: {product['name']}")

# Robust image loader
img_url = str(product.get("image_url", "")).strip()
if img_url:
    try:
        st.image(img_url, width=175)
    except Exception as e:
        if 'SHOW_DEBUG' in globals() and SHOW_DEBUG:
            st.warning(f"Image failed to load: {img_url}\n{e}")
        else:
            st.caption("Image unavailable.")


price_col, info_col = st.columns([2, 1])
with price_col:
    if offered < base:
        st.markdown(
            f"<div style='font-size: 1.8rem;'><s>€{base:.2f}</s> <strong>€{offered:.2f}</strong></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='font-size: 1.8rem;'><strong>€{offered:.2f}</strong></div>",
            unsafe_allow_html=True,
        )
    if SHOW_DEBUG:
        st.caption(f"Base price: €{base:.2f}")

with info_col:
    if SHOW_DEBUG:
        st.caption(f"Session: `{st.session_state.session_id[:8]}`")
        st.caption(f"Current multiplier: {st.session_state.multiplier:.3f}")


def on_change_fair():
    if st.session_state.touched_fair:
        st.session_state.fair_changes += 1
    else:
        st.session_state.touched_fair = True

def on_change_buy():
    if st.session_state.touched_buy:
        st.session_state.buy_changes += 1
    else:
        st.session_state.touched_buy = True

# Prominent main question heading before radio
st.markdown(
    "<div class='big-q'>Do you consider this price to be fair? If you needed this item, would you buy it at this price?</div>",
    unsafe_allow_html=True,
)
combined_answer = st.radio(
    "",
    options=["Yes", "No"],
    index=None,                 # no preselection; first click won't count as a "change"
    horizontal=True,
    key=f"fair_{idx}",
    on_change=on_change_fair,
    label_visibility="collapsed",
)


submitted = st.button("Submit response", use_container_width=True)

if submitted:
    # Determine outcomes
    fair_bool = (combined_answer == "Yes")
    buy_bool = (combined_answer == "Yes")
    bought = (fair_bool and buy_bool)

    revenue = offered if bought else 0.0
    lost = 0.0 if bought else offered

    # Decision time (ms)
    decision_ms = int((time.time() - st.session_state.start_ts) * 1000)
    fair_changes = int(st.session_state.fair_changes)
    buy_changes = int(st.session_state.buy_changes)

    # Record history
    st.session_state.history.append({
        "product_id": product["id"],
        "product_name": product["name"],
        "base_price": base,
        "offered_price": offered,
        "fair": bool(fair_bool),
        "would_buy": bool(buy_bool),
        "bought": bool(bought),
        "revenue": round_price_eur(revenue),
        "lost_revenue": round_price_eur(lost),
        "decision_ms": decision_ms,
        "fair_changes": fair_changes,
        "buy_changes": buy_changes,
    })

    # Adapt price for NEXT product (time-aware)
    # Decide if this was a "slow" decision: either took long OR user flipped answers a lot
    total_flips = int(st.session_state.get("fair_changes", 0)) + int(st.session_state.get("buy_changes", 0))
    is_slow = (int(st.session_state.get("decision_ms", 999999)) >= FAST_DECISION_MS) or (total_flips >= HESITATION_MIN_FLIPS)

    if bought:
        step = UP_SLOW if is_slow else UP_FAST
        st.session_state.multiplier = clamp(st.session_state.multiplier * (1 + step),
                                            MIN_MULTIPLIER, MAX_MULTIPLIER)
    else:
        step = DOWN_SLOW if is_slow else DOWN_FAST
        st.session_state.multiplier = clamp(st.session_state.multiplier * (1 - step),
                                            MIN_MULTIPLIER, MAX_MULTIPLIER)

    # Go to next product or finish
    st.session_state.idx += 1
    if st.session_state.idx >= len(st.session_state.products):
        st.session_state.finished = True

    st.rerun()
