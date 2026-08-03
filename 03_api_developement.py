import streamlit as st
import requests
 

# Config
API_URL = "http://127.0.0.1:8000/predict"
 
st.set_page_config(
    page_title="Diabetes Risk Indicator",
    page_icon="🩺",
    layout="centered",
)



# Header
st.title("🩺 Diabetes Risk Indicator")
st.write(
    "This tool estimates diabetes risk category based on self-reported "
    "lifestyle and health factors, using a model trained on CDC BRFSS survey "
    "data. **This is not a medical diagnosis** — it's an educational, "
    "awareness-focused estimate. Please consult a healthcare professional "
    "for an accurate assessment."
)
st.divider()
 


# Input form
st.subheader("Tell us about yourself")
with st.form("risk_form"):
    col1, col2 = st.columns(2)
 
    with col1:
        high_bp = st.selectbox(
            "Have you been told you have high blood pressure?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        high_chol = st.selectbox(
            "Have you been told you have high cholesterol?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        heart_disease = st.selectbox(
            "Have you ever been diagnosed with heart disease or had a heart attack?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        diff_walk = st.selectbox(
            "Do you have serious difficulty walking or climbing stairs?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
 
    with col2:
        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=10.0, max_value=80.0, value=25.0, step=0.1,
            help="Don't know your BMI? weight(kg) / height(m)^2",
        )
        gen_hlth = st.slider(
            "General health (1 = Excellent, 5 = Poor)",
            min_value=1, max_value=5, value=3,
        )
        phys_hlth = st.slider(
            "In the past 30 days, how many days was your physical health not good?",
            min_value=0, max_value=30, value=0,
        )
        age_bracket = st.selectbox(
            "Age range",
            options=list(range(1, 14)),
            format_func=lambda x: {
                1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
                6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
                11: "70-74", 12: "75-79", 13: "80+",
            }[x],
        )
 
    submitted = st.form_submit_button("Get My Risk Assessment", use_container_width=True)



# Handle submission
if submitted:
    payload = {
        "HighBP": high_bp,
        "HighChol": high_chol,
        "BMI": bmi,
        "HeartDiseaseorAttack": heart_disease,
        "GenHlth": gen_hlth,
        "PhysHlth": phys_hlth,
        "DiffWalk": diff_walk,
        "Age": age_bracket,
    }
 
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
 
        st.divider()
        st.subheader("Your Result")
 
        label = result["predicted_label"]
        probs = result["probabilities"]
 
        label_display = {
            "non-diabetic": ("🟢", "Lower estimated risk"),
            "pre-diabetic": ("🟡", "Elevated estimated risk"),
            "diabetic": ("🔴", "Higher estimated risk"),
        }
        icon, description = label_display.get(label, ("⚪", label))
 
        st.markdown(f"### {icon} {description}")
        st.caption(f"Model prediction: **{label}**")
 
        st.write("**Estimated probability by category:**")
        st.bar_chart(probs)
 
        st.info(result["disclaimer"])
 
    except requests.exceptions.ConnectionError:
        st.error(
            "Couldn't reach the prediction API. Make sure it's running locally "
            "with `uvicorn main:app --reload` from the `api/` folder."
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"The API returned an error: {e}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
 


# Educational resources (static — always shown, no API call needed)
st.divider()
st.subheader("📚 Learn More")
st.write(
    "Whatever your result, here are some reputable resources for learning "
    "more about diabetes risk factors and prevention:"
)
st.markdown(
    """
- [CDC — Diabetes Basics](https://www.cdc.gov/diabetes/basics/index.html)
- [CDC — Prediabetes: Your Chance to Prevent Type 2 Diabetes](https://www.cdc.gov/diabetes/basics/prediabetes.html)
- [American Diabetes Association — Risk Test](https://diabetes.org/about-diabetes/risk-test)
"""
)
 
st.caption(
    "Model: Logistic Regression trained on CDC BRFSS 2015 survey data "
    "(UC Irvine Machine Learning Repository). This is a portfolio/educational "
    "project and is not affiliated with the CDC."
)
