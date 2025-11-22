import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="FinScribe – Explainable Categorization", layout="wide")

# ======================
# Sidebar (Admin + Demo)
# ======================
st.sidebar.title("⚙️ Admin & Demo Tools")

if st.sidebar.button("🔄 Reload Taxonomy"):
    try:
        resp = requests.post(f"{API_URL}/reload_taxonomy")
        if resp.status_code == 200:
            data = resp.json()
            st.sidebar.success(f"Reloaded taxonomy. Total categories: {data['num_categories']}")
        else:
            st.sidebar.error(f"Error: {resp.text}")
    except Exception as e:
        st.sidebar.error(f"Failed to reach API: {e}")

if st.sidebar.button("📂 Show Categories"):
    try:
        resp = requests.get(f"{API_URL}/categories")
        if resp.status_code == 200:
            cats = resp.json()["categories"]
            st.sidebar.write("**Category ID → Label**")
            for cid, label in cats.items():
                st.sidebar.write(f"- `{cid}` → {label}")
        else:
            st.sidebar.error(f"Error: {resp.text}")
    except Exception as e:
        st.sidebar.error(f"Failed to reach API: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Noisy Input Demo")

noisy_examples = {
    "Amazon typo": "AMZN MKTP ORD 99X",
    "Starbucks typo": "Starbuxs Mombay",
    "Uber typo": "UBR TRP BLR",
}

selected_noisy = st.sidebar.selectbox("Pick a noisy example:", list(noisy_examples.keys()))
if st.sidebar.button("Use noisy example"):
    st.session_state["input_text"] = noisy_examples[selected_noisy]

# ======================
# Main UI
# ======================
st.title("💸 FinScribe – Explainable Financial Transaction Categorization")

st.markdown(
    "Enter a raw transaction string (e.g., `AMZN MKTPL 1234`, `STARBUCKS MUMBAI`) and see how "
    "FinScribe categorizes it with confidence, exemplars, and a human-in-the-loop learning loop."
)

col_input, col_result = st.columns([1, 2])

# Ensure default text in state
if "input_text" not in st.session_state:
    st.session_state["input_text"] = "AMZN MKTPLACE ORDER 123"

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

if "before_feedback" not in st.session_state:
    st.session_state["before_feedback"] = None

with col_input:
    text = st.text_area(
        "Transaction Text",
        value=st.session_state["input_text"],
        height=120,
        placeholder="Enter transaction description...",
        key="main_text_area",
    )

    def do_predict(store_before=False):
        if not text.strip():
            st.warning("Please enter a transaction string.")
            return
        try:
            with st.spinner("Contacting FinScribe API..."):
                resp = requests.post(f"{API_URL}/predict", json={"text": text})
            if resp.status_code != 200:
                st.error(f"Error: {resp.text}")
                return
            data = resp.json()
            if store_before:
                st.session_state["before_feedback"] = data
            st.session_state["last_prediction"] = data
            st.session_state["input_text"] = text
        except Exception as e:
            st.error(f"Failed to reach API: {e}")

    if st.button("🔍 Predict Category"):
        # store this as "before feedback" for WOW demo
        do_predict(store_before=True)

    if st.button("🔁 Predict Again (after feedback)"):
        # predict again and compare with before_feedback
        do_predict(store_before=False)


with col_result:
    pred = st.session_state.get("last_prediction")
    if pred:
        st.subheader("Prediction")

        badge_color = "red" if pred["needs_review"] else "green"
        st.markdown(
            f"**Predicted Category:** `{pred['prediction']}` "
            f"(Confidence: **{pred['confidence']:.2f}**, "
            f"<span style='color:{badge_color}'>needs_review={pred['needs_review']}</span>)",
            unsafe_allow_html=True,
        )

        # WOW moment: before vs after feedback
        before = st.session_state.get("before_feedback")
        if before and before is not pred:
            st.markdown("#### 🔄 Learning Effect (Before vs After Feedback)")
            st.write(
                f"- **Before feedback:** `{before['prediction']}` "
                f"(confidence **{before['confidence']:.2f}**)"
            )
            st.write(
                f"- **After feedback:** `{pred['prediction']}` "
                f"(confidence **{pred['confidence']:.2f}**)"
            )

        st.markdown("### Top-3 Candidate Categories")
        for c in pred["top_3"]:
            st.write(f"- `{c['category_label']}` — score: **{c['score']:.2f}**")

        st.markdown("### Nearest Exemplars (Why this decision?)")
        if pred["exemplars"]:
            for ex in pred["exemplars"]:
                st.write(
                    f"- `{ex['text']}` → `{ex['category_label']}` "
                    f"(similarity: **{ex['similarity']:.2f}**)"
                )
        else:
            st.caption("No exemplars yet. The model is relying mainly on base categories.")

        st.markdown("### Explanation Terms")
        st.write(", ".join(pred["explanation_terms"]))

        st.markdown("---")
        st.subheader("🔁 Provide Feedback (Human-in-the-loop)")

        fb_col1, fb_col2 = st.columns([2, 1])
        with fb_col1:
            corrected_category_id = st.text_input(
                "Correct category_id (as defined in categories.yaml)",
                value=pred.get("category_id", ""),
                key="fb_category_id",
            )
        with fb_col2:
            if st.button("✅ Submit Feedback"):
                if not corrected_category_id.strip():
                    st.warning("Please enter a category_id.")
                else:
                    try:
                        fb_resp = requests.post(
                            f"{API_URL}/feedback",
                            json={"text": text, "category_id": corrected_category_id.strip()},
                        )
                        if fb_resp.status_code == 200:
                            st.success("Feedback recorded. Model will use this as a new exemplar.")
                        else:
                            st.error(f"Feedback error: {fb_resp.text}")
                    except Exception as e:
                        st.error(f"Failed to reach API: {e}")
    else:
        st.info("Run a prediction to see results here.")
