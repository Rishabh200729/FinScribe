import requests
import streamlit as st
import pandas as pd
import time

# --- CONFIGURATION ---
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="FinScribe AI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎨 CUSTOM CSS (FINTECH DARK MODE) ---
def apply_styling():
    st.markdown("""
        <style>
        /* Global Reset */
        .stApp {
            background-color: #0f172a; /* Dark Navy */
            color: #e2e8f0;
        }
        
        /* Typography */
        h1, h2, h3 { font-family: 'Inter', sans-serif; color: #ffffff !important; }
        p, label, .stMarkdown { color: #cbd5e1 !important; }
        
        /* Cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        
        /* Metrics & Stats */
        div[data-testid="stMetricValue"] {
            color: #10b981 !important; /* Finance Green */
            font-weight: 700;
        }
        div[data-testid="stMetricLabel"] { color: #94a3b8; }

        /* Inputs */
        .stTextArea textarea, .stTextInput input {
            background-color: #0f172a !important;
            border: 1px solid #475569 !important;
            color: white !important;
            border-radius: 8px;
        }
        .stTextArea textarea:focus { border-color: #10b981 !important; }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        
        /* Secondary Buttons (Outline) */
        div[data-testid="stSidebar"] div.stButton > button {
            background: transparent;
            border: 1px solid #475569;
            color: #cbd5e1;
        }
        div[data-testid="stSidebar"] div.stButton > button:hover {
            border-color: #10b981;
            color: #10b981;
        }

        /* Explanation Tags */
        .term-tag {
            background-color: #334155;
            color: #38bdf8;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.85rem;
            margin-right: 5px;
            border: 1px solid #475569;
            display: inline-block;
        }
        
        /* Status Badges */
        .status-badge-review {
            background-color: #7f1d1d; color: #fca5a5;
            padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;
        }
        .status-badge-ok {
            background-color: #064e3b; color: #6ee7b7;
            padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;
        }
        </style>
    """, unsafe_allow_html=True)

apply_styling()

# --- STATE MANAGEMENT ---
if "input_text" not in st.session_state:
    st.session_state["input_text"] = "AMZN MKTPLACE ORDER 123"
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "before_feedback" not in st.session_state:
    st.session_state["before_feedback"] = None

# --- HELPER FUNCTIONS ---
def do_predict(text, store_before=False):
    if not text.strip():
        st.warning("⚠️ Please enter a transaction string.")
        return
    try:
        with st.spinner("🧠 Analyzing Transaction Patterns..."):
            # Artificial delay for UI effect (optional, remove in prod)
            # time.sleep(0.3) 
            resp = requests.post(f"{API_URL}/predict", json={"text": text})
        
        if resp.status_code != 200:
            st.error(f"API Error: {resp.text}")
            return
            
        data = resp.json()
        if store_before:
            st.session_state["before_feedback"] = data
        st.session_state["last_prediction"] = data
        st.session_state["input_text"] = text
        
    except Exception as e:
        st.error(f"Connection Error: {e}")

# ======================
# SIDEBAR (ADMIN)
# ======================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2168/2168709.png", width=50)
    st.title("FinScribe Admin")
    st.markdown("<div style='margin-bottom: 20px; color: #64748b;'>v0.2.0 | Hybrid-RAG Engine</div>", unsafe_allow_html=True)

    with st.expander("🧪 Test Scenarios (Noisy Data)", expanded=True):
        noisy_examples = {
            "Amazon (Messy)": "AMZN MKTP ORD 99X",
            "Starbucks (Typo)": "Starbuxs Mombay",
            "Uber (Abbr)": "UBR TRP BLR",
            "Shell (Clean)": "SHELL FUEL STN",
            "Netflix (Sub)": "NTFLX.COM PREM"
        }
        selected_noisy = st.selectbox("Select Example:", list(noisy_examples.keys()), label_visibility="collapsed")
        if st.button("Load Example"):
            st.session_state["input_text"] = noisy_examples[selected_noisy]
            st.rerun()

    st.markdown("---")
    
    with st.expander("⚙️ System Controls"):
        if st.button("🔄 Reload Taxonomy"):
            try:
                resp = requests.post(f"{API_URL}/reload_taxonomy")
                if resp.status_code == 200:
                    st.success(f"Taxonomy Updated! ({resp.json()['num_categories']} Cats)")
                else:
                    st.error("Update Failed")
            except:
                st.error("API Offline")

        if st.button("📂 View Taxonomy"):
            try:
                resp = requests.get(f"{API_URL}/categories")
                cats = resp.json()["categories"]
                st.json(cats)
            except:
                st.error("API Offline")

# ======================
# MAIN HEADER
# ======================
c1, c2 = st.columns([3, 1])
with c1:
    st.title("💸 FinScribe AI")
    st.markdown("### Intelligent Transaction Categorization Engine")
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    # Connection Status Badge
    try:
        if requests.get(f"{API_URL}/health", timeout=1).status_code == 200:
            st.markdown("<div style='text-align:right'><span class='status-badge-ok'>● SYSTEM ONLINE</span></div>", unsafe_allow_html=True)
    except:
        st.markdown("<div style='text-align:right'><span class='status-badge-review'>● SYSTEM OFFLINE</span></div>", unsafe_allow_html=True)

st.markdown("---")

# ======================
# INPUT SECTION
# ======================
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    with st.container():
        st.markdown("#### 📝 Transaction Input")
        text_input = st.text_area(
            "Raw String",
            value=st.session_state["input_text"],
            height=150,
            label_visibility="collapsed",
            key="main_text_area"
        )
        
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🔍 Analyze", use_container_width=True):
                do_predict(text_input, store_before=True)
        with c_btn2:
            if st.button("🔁 Re-Verify", use_container_width=True, help="Run this after submitting feedback to see the fix"):
                do_predict(text_input, store_before=False)

        st.info("💡 Tip: Enter raw bank statement text like 'POS W/D STARBUCKS NEW YORK' to test noise resilience.")

# ======================
# RESULTS SECTION
# ======================
with col_right:
    pred = st.session_state.get("last_prediction")
    
    if not pred:
        # Placeholder State
        st.markdown("""
            <div style='text-align: center; padding: 50px; color: #475569; border: 2px dashed #334155; border-radius: 12px;'>
                <h3>👋 Ready to Analyze</h3>
                <p>Enter a transaction string to see the AI categorization engine in action.</p>
            </div>
        """, unsafe_allow_html=True)
    
    else:
        # --- 1. HERO CARD (PREDICTION) ---
        with st.container():
            # Determine Color & Status
            is_low_conf = pred["needs_review"]
            status_color = "#ef4444" if is_low_conf else "#10b981" # Red vs Green
            status_text = "NEEDS REVIEW" if is_low_conf else "CONFIRMED"
            status_class = "status-badge-review" if is_low_conf else "status-badge-ok"
            
            # Header Row
            r1, r2 = st.columns([3, 1])
            with r1:
                st.caption("PREDICTED CATEGORY")
                st.markdown(f"<h2 style='color: {status_color}; margin:0'>{pred['prediction']}</h2>", unsafe_allow_html=True)
                st.markdown(f"<span class='{status_class}'>{status_text}</span>", unsafe_allow_html=True)
            with r2:
                st.metric("Confidence", f"{pred['confidence']:.2f}")

            st.markdown("---")
            
            # --- 2. LEARNING EFFECT (DIFF) ---
            before = st.session_state.get("before_feedback")
            # Only show if we have a 'before' state and the prediction changed OR confidence changed significantly
            if before and (before['category_id'] != pred['category_id'] or abs(before['confidence'] - pred['confidence']) > 0.01):
                st.markdown("#### 🔄 Learning Impact")
                d1, d2, d3 = st.columns(3)
                d1.metric("Previous Prediction", before['prediction'])
                d2.metric("Previous Confidence", f"{before['confidence']:.2f}")
                delta = pred['confidence'] - before['confidence']
                d3.metric("Confidence Boost", f"{delta:+.2f}", delta_color="normal")
                st.markdown("---")

            # --- 3. EXPLAINABILITY ---
            e1, e2 = st.columns(2)
            
            with e1:
                st.markdown("#### 📊 Top Candidates")
                for c in pred["top_3"]:
                    # Custom Progress Bar
                    st.write(f"**{c['category_label']}**")
                    # Ensure score is float and clamped between 0.0 and 1.0
                    score_val = float(c['score'])
                    score_val = max(0.0, min(1.0, score_val)) 
                    st.progress(score_val)
            
            with e2:
                exp = pred.get("explanation", {})

                st.markdown("#### 🧠 Explainability")

                # 1. Simple reason summary
                friendly = exp.get("friendly_summary", {})
                st.write(
                    f"**Why this was predicted:** "
                    f"{friendly.get('semantic', '')}, "
                    f"{friendly.get('exemplar', '')}, "
                    f"{friendly.get('fuzzy', '')}."
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # 2. Show top 2 important tokens
                tokens = exp.get("token_importance", [])
                if tokens:
                    st.caption("Key tokens the model focused on:")
                    simple_tokens = [t['token'] for t in tokens][:3] 
                    token_tags = "".join(
                        [f"<span class='term-tag'>{t}</span>" for t in simple_tokens]
                    )
                    st.markdown(token_tags, unsafe_allow_html=True)
                else:
                    st.caption("No specific tokens identified.")

                st.markdown("<br>", unsafe_allow_html=True)

                # 3. Show 1 nearest exemplar
                nearest = exp.get("nearest_exemplars", [])
                if nearest:
                    top_ex = nearest[0]
                    st.caption("Closest matching past example:")
                    st.write(f"`{top_ex['exemplar_text']}` → **{top_ex['exemplar_category']}**")
                else:
                    st.caption("No exemplar match found.")


        # --- 4. FEEDBACK LOOP ---
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            st.markdown("### 👩‍🏫 Teach the AI (Feedback)")
            st.caption("If the prediction is wrong, correct it below. The system will learn instantly.")
            
            f1, f2 = st.columns([3, 1], vertical_alignment="bottom")
            with f1:
                correct_cat = st.text_input("Correct Category ID", value=pred.get("category_id", ""), help="Enter the ID from categories.yaml (e.g., food_coffee)")
            with f2:
                if st.button("✅ Submit Fix", use_container_width=True):
                    if not correct_cat.strip():
                        st.toast("⚠️ Please enter a category ID")
                    else:
                        try:
                            fb_resp = requests.post(
                                f"{API_URL}/feedback",
                                json={"text": st.session_state["input_text"], "category_id": correct_cat.strip()},
                            )
                            if fb_resp.status_code == 200:
                                st.toast("🎉 Feedback Learned! Re-verify to see changes.", icon="✅")
                            else:
                                st.toast(f"Error: {fb_resp.text}", icon="❌")
                        except Exception as e:
                            st.error(f"API Error: {e}")