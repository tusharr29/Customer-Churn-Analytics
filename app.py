
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E8EDF3;
}

h1, h2, h3 {
    color: #172B4D;
    letter-spacing: -0.5px;
}

p, label {
    color: #526174;
}

.metric-box {
    background: #FFFFFF;
    border: 1px solid #E5EAF0;
    border-radius: 12px;
    padding: 20px;
    min-height: 115px;
}

.metric-title {
    font-size: 13px;
    color: #718096;
    font-weight: 500;
}

.metric-number {
    font-size: 27px;
    color: #172B4D;
    font-weight: 700;
    margin-top: 10px;
}

.info-panel {
    background: #F5F8FC;
    border: 1px solid #E5EAF0;
    border-radius: 12px;
    padding: 22px;
}

.result-panel {
    background: #F8FAFC;
    border: 1px solid #E5EAF0;
    border-radius: 12px;
    padding: 24px;
}

.small-text {
    font-size: 13px;
    color: #718096;
}

div.stButton > button {
    border-radius: 8px;
    height: 45px;
    font-weight: 600;
}

div[data-testid="stFormSubmitButton"] button {
    width: 100%;
}

hr {
    border-color: #E8EDF3;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_objects():

    model = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("ordinal_encoder.pkl")

    return model, scaler, encoder


try:
    model, scaler, encoder = load_objects()

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ==========================================
# CONFIGURATION
# ==========================================

THRESHOLD = 0.40

FEATURES = [
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'MonthlyCharges'
]

ENCODER_INDICES = {
    'Partner': 2,
    'Dependents': 3,
    'MultipleLines': 5,
    'InternetService': 6,
    'OnlineSecurity': 7,
    'OnlineBackup': 8,
    'DeviceProtection': 9,
    'TechSupport': 10,
    'StreamingTV': 11,
    'StreamingMovies': 12,
    'Contract': 13,
    'PaperlessBilling': 14,
    'PaymentMethod': 15
}


# ==========================================
# PREPROCESSING
# ==========================================

def prepare_input(customer):

    encoded = customer.copy()

    for column, index in ENCODER_INDICES.items():

        categories = list(encoder.categories_[index])

        if customer[column] not in categories:
            raise ValueError(
                f"Unknown category for {column}"
            )

        encoded[column] = categories.index(
            customer[column]
        )

    # Use original fitted StandardScaler statistics.
    # Original scaler order:
    # customerID, tenure, MonthlyCharges, TotalCharges

    encoded['tenure'] = (
        customer['tenure'] - scaler.mean_[1]
    ) / scaler.scale_[1]

    encoded['MonthlyCharges'] = (
        customer['MonthlyCharges'] - scaler.mean_[2]
    ) / scaler.scale_[2]

    input_df = pd.DataFrame(
        [[encoded[column] for column in FEATURES]],
        columns=FEATURES
    )

    return input_df


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown("## Churn Analytics")

    st.caption("CUSTOMER INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Predict Churn",
            "Model Insights"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("**MODEL DETAILS**")

    st.caption("Algorithm: K-Nearest Neighbors")
    st.caption("Problem: Binary Classification")
    st.caption("Input Features: 16")
    st.caption("Threshold: 0.40")

    st.divider()

    st.caption("Machine Learning Portfolio Project")


# ==========================================
# HEADER
# ==========================================

st.markdown(
    "<h1 style='font-size:30px;'>Customer Churn Analytics</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color:#718096;'>"
    "Telecom customer retention and churn prediction platform"
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# OVERVIEW PAGE
# ==========================================

if page == "Overview":

    st.subheader("Project Overview")

    st.write(
        "This application uses a trained K-Nearest Neighbors "
        "classification model to estimate the probability "
        "of customer churn based on subscription details, "
        "customer information and service usage."
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">ML Algorithm</div>
            <div class="metric-number">KNN</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">ROC-AUC Score</div>
            <div class="metric-number">79.03%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Model Features</div>
            <div class="metric-number">16</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.subheader("Business Objective")

    st.markdown("""
    The objective is to identify customers who are more likely
    to discontinue their telecom subscriptions.

    Businesses can use these predictions to:

    - Identify customers who may be at risk of leaving.
    - Understand customer subscription patterns.
    - Plan targeted customer retention campaigns.
    - Improve customer retention strategies.
    """)

    st.divider()

    st.subheader("How to use this application")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-panel">
        <b>01. Enter Customer Details</b>
        <p class="small-text">
        Provide customer demographics, subscription,
        internet and billing information.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-panel">
        <b>02. Generate Prediction</b>
        <p class="small-text">
        The trained KNN model estimates the customer's
        probability of churning.
        </p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "Predict Churn":

    st.subheader("Customer Information")

    st.caption(
        "Enter the customer's details to generate a churn prediction."
    )

    with st.form("prediction_form"):

        # CUSTOMER PROFILE

        st.markdown("#### Customer Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with col2:
            partner = st.selectbox(
                "Partner",
                encoder.categories_[2]
            )

        with col3:
            dependents = st.selectbox(
                "Dependents",
                encoder.categories_[3]
            )

        st.divider()

        # SUBSCRIPTION DETAILS

        st.markdown("#### Subscription Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.slider(
                "Tenure (Months)",
                0, 72, 12
            )

        with col2:
            contract = st.selectbox(
                "Contract Type",
                encoder.categories_[13]
            )

        with col3:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=150.0,
                value=70.0,
                step=1.0
            )

        st.divider()

        # SERVICE DETAILS

        st.markdown("#### Internet & Additional Services")

        col1, col2, col3 = st.columns(3)

        with col1:

            multiple_lines = st.selectbox(
                "Multiple Lines",
                encoder.categories_[5]
            )

            internet = st.selectbox(
                "Internet Service",
                encoder.categories_[6]
            )

            security = st.selectbox(
                "Online Security",
                encoder.categories_[7]
            )

            backup = st.selectbox(
                "Online Backup",
                encoder.categories_[8]
            )

        with col2:

            protection = st.selectbox(
                "Device Protection",
                encoder.categories_[9]
            )

            support = st.selectbox(
                "Tech Support",
                encoder.categories_[10]
            )

            tv = st.selectbox(
                "Streaming TV",
                encoder.categories_[11]
            )

            movies = st.selectbox(
                "Streaming Movies",
                encoder.categories_[12]
            )

        with col3:

            paperless = st.selectbox(
                "Paperless Billing",
                encoder.categories_[14]
            )

            payment = st.selectbox(
                "Payment Method",
                encoder.categories_[15]
            )

            st.info(
                "Select the service options that match "
                "the customer's current subscription."
            )

        st.divider()

        submitted = st.form_submit_button(
            "Generate Churn Prediction",
            type="primary"
        )


    # GENERATE PREDICTION

    if submitted:

        customer = {

            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'MultipleLines': multiple_lines,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': protection,
            'TechSupport': support,
            'StreamingTV': tv,
            'StreamingMovies': movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges

        }

        try:

            input_df = prepare_input(customer)

            if input_df.shape[1] != model.n_features_in_:
                raise ValueError("Feature count mismatch.")

            probabilities = model.predict_proba(input_df)[0]

            classes = list(model.classes_)

            churn_probability = probabilities[
                classes.index(1)
            ]

            stay_probability = probabilities[
                classes.index(0)
            ]

            prediction = (
                churn_probability >= THRESHOLD
            )

            st.divider()

            st.subheader("Prediction Results")

            if prediction:

                st.error(
                    "Customer identified as High Churn Risk"
                )

                st.write(
                    "The predicted churn probability is "
                    "above the selected classification threshold."
                )

                st.info(
                    "Business suggestion: Consider a retention "
                    "offer, contract incentives or proactive support."
                )

            else:

                st.success(
                    "Customer identified as Low Churn Risk"
                )

                st.write(
                    "The predicted churn probability is "
                    "below the selected classification threshold."
                )

                st.info(
                    "Business suggestion: Continue monitoring "
                    "customer satisfaction and service quality."
                )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Churn Probability",
                    f"{churn_probability * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Retention Probability",
                    f"{stay_probability * 100:.2f}%"
                )

            st.subheader("Probability Breakdown")

            probability_df = pd.DataFrame(
                {
                    "Outcome": ["Stay", "Churn"],
                    "Probability": [
                        stay_probability,
                        churn_probability
                    ]
                }
            )

            st.bar_chart(
                probability_df.set_index("Outcome")
            )

            with st.expander("View submitted customer information"):

                summary = pd.DataFrame(
                    customer.items(),
                    columns=["Feature", "Value"]
                )

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Predictions are estimates and should not be "
                "treated as guarantees of future customer behaviour."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# ==========================================
# MODEL INSIGHTS PAGE
# ==========================================

elif page == "Model Insights":

    st.subheader("Model Performance")

    st.write(
        "Evaluation metrics from the trained KNN model."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ROC-AUC", "79.03%")

    with col2:
        st.metric("Accuracy at 0.5", "77%")

    with col3:
        st.metric("Selected Threshold", "0.40")

    st.divider()

    st.markdown("#### Classification Performance")

    metrics_df = pd.DataFrame({

        "Metric": [
            "Churn Precision",
            "Churn Recall",
            "Churn F1-score"
        ],

        "Threshold 0.5": [
            0.58, 0.51, 0.54
        ],

        "Threshold 0.4": [
            0.49, 0.74, 0.59
        ]

    })

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Metrics are rounded from the original model evaluation."
    )

    st.divider()

    st.markdown("#### Confusion Matrix at Threshold 0.4")

    confusion = pd.DataFrame(
        [[745, 291], [97, 276]],
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"]
    )

    st.dataframe(
        confusion,
        use_container_width=True
    )

    st.divider()

    st.markdown("#### Classification Threshold")

    st.write(
        "A threshold of 0.40 was selected to increase "
        "the model's ability to identify customers who may churn."
    )

    st.info(
        "At this threshold, churn recall is approximately 74%, "
        "compared with approximately 51% at the 0.50 threshold."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Customer Churn Analytics | Built with Python, "
    "Scikit-learn and Streamlit"
)
