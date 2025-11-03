
# app.py
# Streamlit Adaptive Pricing Experiment (simple, production-ready prototype)
import time
import os
import uuid
import math
import shutil
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image
import io
import base64

import gspread
from google.oauth2.service_account import Credentials

APP_TITLE = "Uber Trip Pricing Experiment"
NUM_PRODUCTS_TO_SHOW = 10         # how many trips per session (5 base + 5 repeats)
UP_PCT = 0.12                     # +12% if ( would buy)
DOWN_PCT = 0.08                   # -8% otherwise (wouldn't buy)
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

@st.cache_data
def load_products():
    """Load trips from CSV or fallback to defaults."""
    if os.path.exists(PRODUCTS_CSV):
        df = pd.read_csv(PRODUCTS_CSV)

        # Basic validation
        if "name" not in df.columns or "base_price" not in df.columns:
            st.error("products.csv must include at least 'name' and 'base_price' columns.")
            st.stop()

        # Fill / normalize columns
        if "id" not in df.columns:
            df["id"] = [f"t{i+1}" for i in range(len(df))]
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
        {"id": "t1", "name": "Marquês to Campo grande", "base_price": 4.0,
         "image_url": ""},
        {"id": "t2", "name": "Marquês to Cais do Sodré", "base_price": 6.0,
         "image_url": ""},
        {"id": "t3", "name": "Marquês to Belém", "base_price": 9.0,
         "image_url": ""},
        {"id": "t4", "name": "Marquês to Carcavelos", "base_price": 16.0,
         "image_url": ""},
        {"id": "t5", "name": "Marquês to Costa da Caparica", "base_price": 19.0,
         "image_url": ""},
    ]
    

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def round_price_eur(x):
    """Round to 2 decimals for EUR."""
    return float(f"{x:.2f}")

def ensure_csv(path, columns):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    else:
        # Check if existing CSV has the same columns
        try:
            existing_df = pd.read_csv(path, nrows=0)  # Read only header
            existing_cols = list(existing_df.columns)
            if existing_cols != columns:
                # Schema changed - backup old file and create new one with correct schema
                backup_path = path.replace('.csv', '_backup.csv')
                if os.path.exists(path):
                    shutil.copy(path, backup_path)
                # Create new CSV with correct schema
                pd.DataFrame(columns=columns).to_csv(path, index=False)
        except Exception:
            # If any error reading, recreate the file
            pd.DataFrame(columns=columns).to_csv(path, index=False)

def append_rows(path, rows, columns):
    ensure_csv(path, columns)
    df = pd.DataFrame(rows, columns=columns)
    # mode='a' appends; header=False to avoid repeating header
    df.to_csv(path, mode="a", header=False, index=False)

@st.cache_data(ttl=60)  # Cache for 60 seconds
def read_tx():
    cols = [
        "timestamp","session_id","product_id","product_name","base_price",
        "offered_price","fair","would_buy","bought","revenue","lost_revenue",
        "decision_ms","fair_changes","buy_changes","up_pct","down_pct","others_avg_paid"
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
              "up_pct", "down_pct", "fair_changes", "buy_changes", "others_avg_paid"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ensure expected columns exist even if skipped lines dropped some
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    # keep only known columns to avoid downstream mismatches
    df = df[cols]

    return df

@st.cache_data(ttl=60)  # Cache for 60 seconds
def others_avg_paid_by_product(exclude_session_id):
    # Try to read from Google Sheets first, then fallback to CSV
    df = None
    
    # Try Google Sheets first
    try:
        ws_tx, _ = get_sheets_client()
        if ws_tx:
            # Read all data from Google Sheets
            records = ws_tx.get_all_records()
            if records:
                df = pd.DataFrame(records)
                # Normalize column names if needed
                if 'product_id' in df.columns and 'offered_price' in df.columns:
                    # Convert types
                    df['offered_price'] = pd.to_numeric(df['offered_price'], errors='coerce')
                    if 'bought' in df.columns:
                        df['bought'] = df['bought'].astype(str).str.lower().isin(['true', '1', 'yes'])
    except Exception:
        # Silently fallback to CSV
        pass
    
    # Fallback to CSV if Sheets didn't work or returned empty
    if df is None or df.empty:
        df = read_tx()
    
    if df.empty:
        return {}
    
    # Exclude current session
    if 'session_id' not in df.columns:
        return {}
    
    df_excl = df[df["session_id"] != exclude_session_id]
    if df_excl.empty:
        return {}

    # Ensure we have the required columns
    if 'product_id' not in df_excl.columns or 'offered_price' not in df_excl.columns:
        return {}

    # Primary: average among purchases only
    if 'bought' in df_excl.columns:
        bought_mask = df_excl["bought"] == True
        df_bought = df_excl[bought_mask]
        
        if not df_bought.empty:
            bought_means = df_bought.groupby("product_id")["offered_price"].mean().to_dict()
        else:
            bought_means = {}
    else:
        bought_means = {}

    # Fallback: if a product has no purchases, compute average of all shown prices
    all_means = df_excl.groupby("product_id")["offered_price"].mean().to_dict()

    # Merge: prefer bought_means, else fallback to all_means
    result = {}
    # Include all product_ids that exist
    all_product_ids = set(all_means.keys()) | set(bought_means.keys())
    for pid in all_product_ids:
        val = bought_means.get(pid, all_means.get(pid, None))
        if val is not None and not pd.isna(val):
            result[str(pid)] = float(val)
    
    return result
@st.cache_resource
def get_sheets_client():
    try:
        sa_info = st.secrets["gcp_service_account"]
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(st.secrets["sheets"]["sheet_id"])
        ws_tx = sh.worksheet("transactions")
        ws_sv = sh.worksheet("surveys")
        return ws_tx, ws_sv
    except Exception as e:
        if 'SHOW_DEBUG' in globals() and SHOW_DEBUG:
            st.warning(f"Google Sheets not configured: {e}")
        return None, None

def ensure_headers(ws, headers):
    try:
        first = ws.row_values(1)
        if not first:
            ws.append_row(headers, value_input_option="USER_ENTERED")
        elif first != headers:
            ws.update('1:1', [headers])
    except Exception:
        pass


# ---------- SESSION STATE ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "products" not in st.session_state:
    all_products = load_products()
    # Build trip sequence with repeats inserted after their originals
    products_list = []
    
    # Trip 1: Campo grande
    if len(all_products) > 0:
        products_list.append(all_products[0])
        # Immediately add repeat with late night scenario
        repeat_campo = all_products[0].copy()
        repeat_campo["id"] = "t1_night"
        repeat_campo["name"] = "(1 AM) " + repeat_campo["name"]
        repeat_campo["scenario"] = "night"
        repeat_campo["original_idx"] = 0  # Index in products_list where original appears
        products_list.append(repeat_campo)
    
    # Trip 2: Cais do Sodré
    if len(all_products) > 1:
        cais_idx = len(products_list)  # Remember where Cais appears
        products_list.append(all_products[1])
        # Immediately add repeat with rain scenario
        repeat_cais = all_products[1].copy()
        repeat_cais["id"] = "t2_rain"
        repeat_cais["name"] = "(rainy day) " + repeat_cais["name"]
        repeat_cais["scenario"] = "rain"
        repeat_cais["original_idx"] = cais_idx  # Index in products_list where original appears
        products_list.append(repeat_cais)
    
    # Trip 3: Belém
    if len(all_products) > 2:
        belem_idx = len(products_list)  # Remember where Belém appears
        products_list.append(all_products[2])
        # Immediately add repeat with traffic scenario
        repeat_belem = all_products[2].copy()
        repeat_belem["id"] = "t3_traffic"
        repeat_belem["name"] = "(peak hour) " + repeat_belem["name"]
        repeat_belem["scenario"] = "traffic"
        repeat_belem["original_idx"] = belem_idx  # Index in products_list where original appears
        products_list.append(repeat_belem)
    
    # Trip 4: Carcavelos
    if len(all_products) > 3:
        carcavelos_idx = len(products_list)  # Remember where Carcavelos appears
        products_list.append(all_products[3])
        # Immediately add repeat with special event scenario
        repeat_carcavelos = all_products[3].copy()
        repeat_carcavelos["id"] = "t4_event"
        repeat_carcavelos["name"] = "(special event) " + repeat_carcavelos["name"]
        repeat_carcavelos["scenario"] = "event"
        repeat_carcavelos["original_idx"] = carcavelos_idx  # Index in products_list where original appears
        products_list.append(repeat_carcavelos)
    
    # Trip 5: Costa da Caparica
    if len(all_products) > 4:
        caparica_idx = len(products_list)  # Remember where Costa da Caparica appears
        products_list.append(all_products[4])
        # Immediately add repeat with public transport strike scenario
        repeat_caparica = all_products[4].copy()
        repeat_caparica["id"] = "t5_strike"
        repeat_caparica["name"] = "(public transport strike) " + repeat_caparica["name"]
        repeat_caparica["scenario"] = "strike"
        repeat_caparica["original_idx"] = caparica_idx  # Index in products_list where original appears
        products_list.append(repeat_caparica)
    
    st.session_state.products = products_list

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
st.set_page_config(page_title=APP_TITLE, page_icon="🚗", layout="centered")
st.title(APP_TITLE)
st.caption("This behavioral experiment was conducted as part of a Master’s thesis and involves presenting participants with the prices of five different trips departing from Marquês de Pombal.")
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


SHOW_DEBUG = False  # Set to True to show admin panel and debug info

# If we've just finished the survey, show a lightweight close page
if st.session_state.get("show_close", False):
    st.header("Thank you!")
    st.success("Your responses were saved successfully. You can now close this tab.")
    st.stop()

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
    /* Improve image quality and rendering for photos */
    img { 
        image-rendering: -webkit-optimize-contrast;
        image-rendering: high-quality;
        max-width: 100%;
        height: auto;
        object-fit: contain;
    }
    /* Ensure Streamlit images maintain quality */
    [data-testid="stImage"] img {
        image-rendering: -webkit-optimize-contrast;
        image-rendering: high-quality;
    }
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
        # Compute "others paid" averages
        others = others_avg_paid_by_product(st.session_state.session_id)
        
        # Debug: show what we found
        if SHOW_DEBUG:
            st.write(f"Debug: Session ID: {st.session_state.session_id}")
            st.write(f"Debug: Found {len(others)} products with others' data: {others}")
            st.write(f"Debug: Current product IDs: {list(hist_df['product_id'].unique())}")
        
        # Map product_id to others_avg_paid, converting to float and handling None
        hist_df["others_avg_paid"] = hist_df["product_id"].map(others)
        # Convert None/NaN to pd.NA but keep numeric values as-is
        hist_df["others_avg_paid"] = hist_df["others_avg_paid"].apply(
            lambda x: float(x) if pd.notna(x) and x is not None else pd.NA
        )

        # Show what you paid vs others
        display_cols = ["product_name","base_price","offered_price","others_avg_paid"]
        st.subheader("What you saw vs. what others paid (avg)")
        st.markdown("""
        <style>
        .dataframe { font-size: 1.6rem !important; }
        .dataframe td { 
            font-size: 1.6rem !important; 
            padding: 0.75rem !important;
            white-space: normal !important;
            word-wrap: break-word !important;
        }
        .dataframe th { 
            font-size: 1.6rem !important; 
            padding: 0.75rem !important;
            white-space: normal !important;
            word-wrap: break-word !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.dataframe(hist_df[display_cols].rename(columns={
            "product_name":"Trip",
            "base_price":"Base €",
            "offered_price":"You saw €",
            "others_avg_paid":"How much others paid, on average"
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
        st.subheader("Perception of Uber dynamic pricing")

        st.markdown("<div class='big-q secondary'>How fair is it for Uber to charge higher prices for higher demand periods, such as peak hours, special events, late night or rainy days?</div>", unsafe_allow_html=True)
        time_based_fairness = st.slider(
            "",
            min_value=0, max_value=10, value=5, help="0 = Not fair at all, 10 = Completely fair",
            label_visibility="collapsed",
            key="time_based_fairness"
        )
        st.markdown("<div style='display: flex; justify-content: space-between; font-size: 0.9rem; color: #666; margin-top: -0.5rem; margin-bottom: 1rem;'><span>Very unfair and unacceptable</span><span>Very fair and acceptable</span></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='big-q secondary'>How fair is it for Uber to charge different prices to different users for the same trip at the same time, based on user-specific data such as your previous purchases and known willingness to pay?</div>", unsafe_allow_html=True)
        personalized_fairness = st.slider(
            "",
            min_value=0, max_value=10, value=5, help="0 = Not fair at all, 10 = Completely fair",
            label_visibility="collapsed",
            key="personalized_fairness"
        )
        st.markdown("<div style='display: flex; justify-content: space-between; font-size: 0.9rem; color: #666; margin-top: -0.5rem; margin-bottom: 1rem;'><span>Very unfair and unacceptable</span><span>Very fair and acceptable</span></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='big-q secondary'>How fair and transparent do you feel the pricing process was overall? Were you able to understand why prices might vary?</div>", unsafe_allow_html=True)
        fairness_transparency_score = st.slider(
            "",
            min_value=0, max_value=10, value=5, help="0 = Not fair/transparent at all, 10 = Very fair and transparent",
            label_visibility="collapsed",
            key="fairness_transparency_score"
        )
        st.markdown("<div style='display: flex; justify-content: space-between; font-size: 0.9rem; color: #666; margin-top: -0.5rem; margin-bottom: 1rem;'><span>Very unfair and unacceptable</span><span>Very fair and acceptable</span></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='big-q secondary'>From a fixed pricing for km (0) to totally dynamic pricing (10), what do you prefer, knowing that you could pay more sometimes but also have discounts?</div>", unsafe_allow_html=True)
        prefer_fixed_pricing = st.slider(
            "",
            min_value=0, max_value=10, value=5, help="0 = Fixed pricing for km, 10 = Totally dynamic pricing",
            label_visibility="collapsed",
            key="prefer_fixed_pricing"
        )
        st.markdown("<div style='display: flex; justify-content: space-between; font-size: 0.9rem; color: #666; margin-top: -0.5rem; margin-bottom: 1rem;'><span>Fixed pricing for km</span><span>Totally dynamic pricing</span></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='big-q secondary'>Do you accept dynamic pricing if the price is explained, making you aware of the reasons for a price raise, such as peak hour?</div>", unsafe_allow_html=True)
        acceptance_with_explanation = st.slider(
            "",
            min_value=0, max_value=10, value=5, help="0 = Do not accept, 10 = Fully accept",
            label_visibility="collapsed",
            key="acceptance_with_explanation"
        )
        st.markdown("<div style='display: flex; justify-content: space-between; font-size: 0.9rem; color: #666; margin-top: -0.5rem; margin-bottom: 1rem;'><span>Do not accept</span><span>Fully accept</span></div>", unsafe_allow_html=True)
        
        comments = st.text_area("Any thoughts on Uber's dynamic pricing, fairness, transparency, or trust?")

        if st.button("Submit survey and save"):
            tx_columns = ["timestamp","session_id","product_id","product_name","base_price",
                          "offered_price","fair","would_buy","bought","revenue","lost_revenue",
                          "decision_ms","fair_changes","buy_changes","up_pct","down_pct","others_avg_paid"]

            survey_columns = ["timestamp","session_id","total_revenue","total_lost",
                              "time_based_fairness","personalized_fairness",
                              "fairness_transparency_score","prefer_fixed_pricing",
                              "acceptance_with_explanation","comments"]

            now = datetime.utcnow().isoformat()

            tx_rows = []
            for row in st.session_state.history:
                # Get others_avg_paid for this product from the already computed hist_df
                matching_row = hist_df[hist_df["product_id"] == row["product_id"]]
                if not matching_row.empty:
                    others_avg = matching_row["others_avg_paid"].iloc[0]
                    # Convert pd.NA or NaN to None for saving, otherwise convert to float
                    try:
                        if pd.isna(others_avg):
                            others_avg = None
                        else:
                            # Convert to float if it's a numeric value
                            others_avg = float(others_avg)
                    except (TypeError, ValueError):
                        others_avg = None
                else:
                    others_avg = None
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
                    DOWN_PCT,
                    others_avg
                ])

            total_revenue = sum(r["revenue"] for r in st.session_state.history) if st.session_state.history else 0.0
            total_lost    = sum(r["lost_revenue"] for r in st.session_state.history) if st.session_state.history else 0.0

            survey_row = [[
                now,
                st.session_state.session_id,
                float(total_revenue),
                float(total_lost),
                time_based_fairness,
                personalized_fairness,
                fairness_transparency_score,
                prefer_fixed_pricing,
                acceptance_with_explanation,
                comments,
            ]]

            ws_tx, ws_sv = get_sheets_client()
            if ws_tx and ws_sv:
                ensure_headers(ws_tx, tx_columns)
                ensure_headers(ws_sv, survey_columns)
                for r in tx_rows:
                    ws_tx.append_row(r, value_input_option="USER_ENTERED")
                ws_sv.append_row(survey_row[0], value_input_option="USER_ENTERED")
                st.success("Thanks! Your responses were saved to Google Sheets.")
            else:
                append_rows(TX_CSV, tx_rows, tx_columns)
                append_rows(SURVEY_CSV, survey_row, survey_columns)
                st.success("Saved locally (CSV). Tip: configure Google Sheets in Secrets for persistent storage.")

            # After saving, show a close page on next rerun
            st.session_state.show_close = True
            st.rerun()

    st.stop()

# ---------- MAIN FLOW: show current trip ----------
products = st.session_state.products
idx = st.session_state.idx
product = products[idx]

# Start timing + reset hesitation counters for this trip
st.session_state.start_ts = time.time()
st.session_state.fair_changes = 0
st.session_state.buy_changes = 0
st.session_state.touched_fair = False
st.session_state.touched_buy = False

base = float(product["base_price"])

# Special pricing for repeat trips
if "scenario" in product:
    # This is a repeat trip - check if user accepted the original
    original_idx = product.get("original_idx", 0)
    if original_idx < len(st.session_state.history):
        original_trip = st.session_state.history[original_idx]
        if original_trip.get("bought", False):
            # User accepted the original price - increase by 15%
            offered = round_price_eur(original_trip["offered_price"] * 1.20)
        else:
            # User didn't accept - keep same price as original
            offered = round_price_eur(original_trip["offered_price"])
    else:
        # Fallback if history not available
        offered = round_price_eur(base * st.session_state.multiplier)
else:
    # Normal trip - use current multiplier
    offered = round_price_eur(base * st.session_state.multiplier)
discount_pct = 0.0
if offered < base:
    discount_pct = (1 - offered / base) * 100.0

# UI
st.progress((idx) / len(products))
st.subheader(f"Trip {idx+1} of {len(products)}: {product['name']}")

# Display trip image if available
img_url = str(product.get("image_url", "")).strip()
if img_url:
    # Try multiple approaches to load the image
    img_loaded = False
    error_msg = None
    
    # First try: PIL with relative path
    if os.path.exists(img_url):
        try:
            img = Image.open(img_url)
            # Resize image for faster loading (max width 800px)
            max_width = 800
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert RGBA/LA/P to RGB for better compatibility
            if img.mode in ('RGBA', 'LA'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = rgb_img
            elif img.mode == 'P':
                img = img.convert('RGB')
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Try base64 encoding approach with JPEG for smaller file size
            try:
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85, optimize=True)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_html = f'<img src="data:image/jpeg;base64,{img_str}" style="max-width:65%;height:auto;display:block;margin:0 auto;">'
                st.markdown(img_html, unsafe_allow_html=True)
                img_loaded = True
            except Exception as e3:
                # Fallback to Streamlit's st.image
                st.image(img, width=500)
                img_loaded = True
        except Exception as e:
            error_msg = str(e)
            # Try direct file path without PIL
            try:
                st.image(img_url, width=500)
                img_loaded = True
            except Exception as e2:
                error_msg = f"PIL: {e}, Direct: {e2}"
    
    # If still not loaded, show detailed error
    if not img_loaded:
        # Temporarily show debug info
        st.error(f"⚠️ Image loading failed")
        st.caption(f"Path: {img_url}")
        st.caption(f"Exists: {os.path.exists(img_url)}")
        if error_msg:
            st.caption(f"Error: {error_msg}")
        if os.path.exists(img_url):
            try:
                test_img = Image.open(img_url)
                st.caption(f"Format: {test_img.format}, Mode: {test_img.mode}, Size: {test_img.size}")
            except Exception as e:
                st.caption(f"PIL test error: {e}")

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
# Check if this is a repeat trip with a scenario
if "scenario" in product:
    if product["scenario"] == "night":
        question_text = "<div class='big-q'>It's 1 in the morning. Do you consider this price to be fair? If you needed this trip, would you book it at this price?</div>"
    elif product["scenario"] == "rain":
        question_text = "<div class='big-q'>Now it's raining. Do you consider this price to be fair? If you needed this trip, would you book it at this price?</div>"
    elif product["scenario"] == "traffic":
        question_text = "<div class='big-q'>It's 7pm and there is heavy traffic. Do you consider this price to be fair? If you needed this trip, would you book it at this price?</div>"
    elif product["scenario"] == "event":
        question_text = "<div class='big-q'>There is a special event and many people are ordering Ubers. Do you consider this price to be fair? If you needed this trip, would you book it at this price?</div>"
    elif product["scenario"] == "strike":
        question_text = "<div class='big-q'>There is a public transport strike. Do you consider this price to be fair? If you needed this trip, would you book it at this price?</div>"
    else:
        question_text = "<div class='big-q'>Do you consider this price to be fair? If you needed this trip, would you book it at this price?</div>"
else:
    question_text = "<div class='big-q'>Do you consider this price to be fair? If you needed this trip, would you book it at this price?</div>"

st.markdown(question_text, unsafe_allow_html=True)
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

    # Go to next trip or finish
    st.session_state.idx += 1
    if st.session_state.idx >= len(st.session_state.products):
        st.session_state.finished = True

    st.rerun()
