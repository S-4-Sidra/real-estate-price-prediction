import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime
import json
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="🏡 Real Estate House Price Prediction", layout="wide")

# -----------------------------
# Session State Defaults
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Welcome"
if "history" not in st.session_state:
    st.session_state.history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# -----------------------------
# Feedback System Functions
# -----------------------------
def load_feedback():
    if os.path.exists("feedback.json"):
        with open("feedback.json", "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def save_feedback(feedback_list):
    with open("feedback.json", "w") as f:
        json.dump(feedback_list, f, indent=4)

# -----------------------------
# UI Helpers
# -----------------------------
def inject_css(dark=False):
    body_bg = "#0b1020" if dark else "#f9f9fb"
    text_color = "#e5e7eb" if dark else "#111827"
    card_bg = "#1f2937" if dark else "white"
    card_shadow = "0 4px 30px rgba(0,0,0,0.45)" if dark else "0 4px 30px rgba(0,0,0,0.12)"
    tab_bg = "#6d28d9" if dark else "#e0f2fe"
    tab_text = "white" if dark else "#0f172a"
    
    st.markdown(f"""
        <style>
        html, body, [class*="css"] {{
            background: {body_bg} !important;
            color: {text_color} !important;
            font-family: 'Segoe UI', sans-serif;
        }}
        .card {{
            background: {card_bg};
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: {card_shadow};
        }}
        .thank-you-card {{
            background: linear-gradient(135deg, #6d28d9, #a855f7);
            color: white;
            padding: 30px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .feedback-card {{
            background: {card_bg};
            padding: 20px;
            border-radius: 16px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        /* Make tabs colorful */
        .stTabs [role="tablist"] button {{
            background: {tab_bg} !important;
            color: {tab_text} !important;
            border-radius: 10px;
            margin-right: 6px;
            padding: 8px 16px;
        }}
        .stTabs [role="tablist"] button[aria-selected="true"] {{
            background: linear-gradient(90deg, #9333ea, #3b82f6) !important;
            color: white !important;
            font-weight: bold;
        }}
        </style>
    """, unsafe_allow_html=True)

def set_matplotlib_theme(dark=False):
    plt.style.use("dark_background" if dark else "default")

# -----------------------------
# Dark Mode Toggle (Top Right)
# -----------------------------
t1, t2 = st.columns([0.75,0.25])
with t2:
    st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)
inject_css(st.session_state.dark_mode)

# -----------------------------
# Welcome Page
# -----------------------------
if st.session_state.page == "Welcome":
    st.markdown(f"""
        <div style="
            background-image: url('https://images.unsplash.com/photo-1600585154340-be6161a56a0c');
            background-size: cover;
            background-position: center;
            padding: 120px 20px;
            border-radius: 20px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 30px rgba(0,0,0,0.45);
        ">
            <h1 style="font-size:55px;">🏡 Real Estate House Price Prediction</h1>
            <h2 style="font-size:22px;">AI-powered insights for smarter property decisions</h2>
            <p>Predict housing prices instantly with advanced machine learning.</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("### 🔎 Why use this app?")
    st.write("- ✅ Predict accurate house prices instantly")
    st.write("- ✅ Explore location-based insights")
    st.write("- ✅ Save and analyze your past predictions")
    st.write("- ✅ Professional tool for real estate buyers & sellers")

    if st.button("🚀 Get Started"):
        st.session_state.logged_in = True
        st.session_state.page = "App"
        st.rerun()

    st.markdown("<div style='text-align:center; color: gray;'>© 2025 Real Estate Predictor | Developed with ❤️ by Sidra</div>", unsafe_allow_html=True)

# -----------------------------
# Main App with Tabs
# -----------------------------
elif st.session_state.logged_in:
    tabs = st.tabs([
        "🏠 Home",
        "📊 Visualization",
        "🗃️ History",
        "🧭 Location Info",
        "ℹ️ About",
        "🎉 Thank You & Feedback",   # Updated tab name
        "🔑 Logout"
    ])

    # Load model
    try:
        model = joblib.load("house_price_model.pkl")
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

    expected_features = getattr(model, "n_features_in_", None)
    feature_names_in = getattr(model, "feature_names_in_", None)

    # -----------------------------
    # HOME TAB
    # -----------------------------
    with tabs[0]:
        st.title("🏠 House Price Prediction")
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Area (sq ft)", 100, 10000, 1500, 100)
            bedrooms = st.number_input("Number of Bedrooms", 1, 10, 3)
            bathrooms = st.number_input("Number of Bathrooms", 1, 5, 2)
            stories = st.number_input("Number of Stories", 1, 5, 2)
            parking = st.number_input("Parking Spaces", 0, 5, 1)
        with col2:
            mainroad = st.selectbox("Main Road Access", ["Yes","No"])
            guestroom = st.selectbox("Guest Room", ["Yes","No"])
            basement = st.selectbox("Basement", ["Yes","No"])
            hotwaterheating = st.selectbox("Hot Water Heating", ["Yes","No"])
            airconditioning = st.selectbox("Air Conditioning", ["Yes","No"])
            prefarea = st.selectbox("Preferred Area", ["Yes","No"])
            furnishingstatus = st.selectbox("Furnishing Status", ["Furnished","Semi-Furnished","Unfurnished"])
            location = st.text_input("Location / City", placeholder="Enter city (e.g. Lahore, Karachi)")

        yn = lambda v: 1 if v=="Yes" else 0
        base_vals = {
            "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "stories": stories, "mainroad": yn(mainroad), "guestroom": yn(guestroom),
            "basement": yn(basement), "hotwaterheating": yn(hotwaterheating),
            "airconditioning": yn(airconditioning), "parking": parking, "prefarea": yn(prefarea)
        }

        def build_features_df():
            if expected_features and expected_features == len(base_vals)+1:
                fs_map = {"Furnished":0,"Semi-Furnished":1,"Unfurnished":2}
                vals = {**base_vals,"furnishingstatus": fs_map[furnishingstatus]}
                X_local = pd.DataFrame([vals])
            else:
                furnished = 1 if furnishingstatus=="Furnished" else 0
                semi = 1 if furnishingstatus=="Semi-Furnished" else 0
                vals = {**base_vals,"furnishingstatus_furnished":furnished,"furnishingstatus_semi-furnished":semi}
                X_local = pd.DataFrame([vals])
            if feature_names_in is not None:
                for c in feature_names_in:
                    if c not in X_local.columns:
                        X_local[c]=0
                X_local = X_local[[c for c in feature_names_in]]
            return X_local

        X = build_features_df()
        if st.button("Predict Price 💰"):
            try:
                pred = model.predict(X)[0]
                st.success(f"Estimated House Price in {location or 'your city'}: **₨ {pred:,.0f}**")
                st.session_state.history.append({
                    "Location": location or "Unknown","Area": area,"Bedrooms": bedrooms,
                    "Bathrooms": bathrooms,"Stories": stories,"Parking": parking,
                    "Predicted Price": f"₨ {pred:,.0f}"
                })
            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")

    # -----------------------------
    # VISUALIZATION TAB
    # -----------------------------
    with tabs[1]:
        st.title("📊 Data Visualization")
        set_matplotlib_theme(st.session_state.dark_mode)
        if hasattr(model,"feature_importances_"):
            st.subheader("🔑 Feature Importance")
            try:
                fi = pd.DataFrame({
                    "Feature": feature_names_in if feature_names_in is not None else X.columns,
                    "Importance": model.feature_importances_
                }).sort_values("Importance",ascending=False)
                st.bar_chart(fi.set_index("Feature"))
            except:
                st.info("Feature importance not available.")
        if st.session_state.history:
            st.subheader("📈 Price vs Area Trend")
            df_hist = pd.DataFrame(st.session_state.history)
            df_hist["Predicted Price"] = df_hist["Predicted Price"].str.replace("₨ ","").str.replace(",","").astype(float)
            plt.figure(figsize=(5,2.5))
            plt.plot(df_hist["Area"], df_hist["Predicted Price"], marker="o", linewidth=1.2)
            plt.xlabel("Area (sq ft)"); plt.ylabel("Price (₨)"); plt.grid(alpha=0.25)
            st.pyplot(plt)
        else:
            st.info("Make a few predictions to see charts here.")

    # -----------------------------
    # HISTORY TAB
    # -----------------------------
    with tabs[2]:
        st.title("🗃️ Prediction History")
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df,use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download History CSV",data=csv,file_name="prediction_history.csv",mime="text/csv")
        else:
            st.info("No predictions yet.")

    # -----------------------------
    # LOCATION TAB
    # -----------------------------
    with tabs[3]:
        st.title("🧭 Location Insights")
        city = st.text_input("Enter a City","Lahore")
        st.write(f"Insights for **{city}**")
        st.map(pd.DataFrame({"lat":[31.5204],"lon":[74.3587]}))

    # -----------------------------
    # ABOUT TAB
    # -----------------------------
    with tabs[4]:
        st.title("ℹ️ About This App")
        st.write("Modern ML tool for property values prediction.")
        st.markdown("""
        - 🏠 Instant house price prediction  
        - 📊 Visualizations  
        - 🗃️ History export  
        - 🧭 Location insights  
        - 🌙 Light/Dark mode  
        """)
        st.subheader("📧 Contact")
        st.write("Developed with ❤️ by **Sidra**")

    # -----------------------------
    # THANK YOU & FEEDBACK TAB
    # -----------------------------
    with tabs[5]:
        st.markdown("""
        <div class="thank-you-card">
            <h1>🎉 Thank You!</h1>
            <p style="font-size: 18px;">Thanks for using Real Estate House Price Predictor.</p>
            <p style="font-size: 18px;">💬 We'd love to hear your feedback about your experience!</p>
            <p>Developed with ❤️ by <b>Sidra</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feedback Form
        st.subheader("💬 Share Your Feedback")
        
        with st.form("feedback_form"):
            name = st.text_input("Your Name (optional)")
            rating = st.slider("Rate your experience", 1, 5, 5)
            feedback = st.text_area("Your Feedback", placeholder="What did you like? How can we improve?")
            submitted = st.form_submit_button("Submit Feedback")
            
            if submitted:
                if feedback.strip():
                    # Load existing feedback
                    feedback_list = load_feedback()
                    
                    # Add new feedback
                    new_feedback = {
                        "name": name if name else "Anonymous",
                        "rating": rating,
                        "feedback": feedback,
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    feedback_list.append(new_feedback)
                    save_feedback(feedback_list)
                    
                    st.session_state.feedback_submitted = True
                    st.success("Thank you for your feedback! 💖")
                else:
                    st.warning("Please share your feedback before submitting.")
        
        # Display previous feedbacks
        st.subheader("📝 What Others Are Saying")
        
        feedback_list = load_feedback()
        
        if feedback_list:
            # Show most recent feedback first
            for i, fb in enumerate(reversed(feedback_list)):
                stars = "⭐" * fb.get("rating", 5)
                
                st.markdown(f"""
                <div class="feedback-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4>{fb.get('name', 'Anonymous')}</h4>
                        <div>{stars}</div>
                    </div>
                    <p>{fb.get('feedback', '')}</p>
                    <small>{fb.get('date', '')}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Show only the 5 most recent feedbacks
                if i >= 4:
                    break
        else:
            st.info("No feedback yet. Be the first to share your experience!")

    # -----------------------------
    # LOGOUT TAB
    # -----------------------------
    with tabs[6]:
        st.title("🔑 Logout")
        if st.button("Confirm Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "Welcome"
            st.rerun()