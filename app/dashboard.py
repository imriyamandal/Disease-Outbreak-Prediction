import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


st.set_page_config(
    page_title="Disease Outbreak Prediction",
    page_icon="🦠",
    layout="wide"
)


st.markdown("""
<style>

.main {
    background-color: #0B1020;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

h1, h2, h3 {
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #1F2937;
    padding: 20px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)


df = pd.read_csv("data/cleaned_dataset.csv")

df['total_cases'] = df['total_cases'].fillna(0)

import os

model_path = os.path.join(
    "models",
    "xgboost_model.pkl"
)

model = joblib.load(model_path)


X = df.drop(
    ['total_cases', 'outbreak'],
    axis=1
)

X = X.select_dtypes(include=np.number)


def run_dashboard():

    st.sidebar.title("🦠 Outbreak Dashboard")

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Prediction", "Analytics"]
    )

    city = st.sidebar.selectbox(
        "Select City",
        sorted(df['city'].unique())
    )

    year = st.sidebar.slider(
        "Select Year",
        int(df['year'].min()),
        int(df['year'].max()),
        int(df['year'].max())
    )

    filtered_df = df[
        (df['city'] == city) &
        (df['year'] == year)
    ]

    st.title("🦠 Disease Outbreak Prediction System")

    st.markdown("""
    AI-powered dashboard for predicting disease outbreaks
    using climate and historical health data.
    """)

    total_cases = int(
        filtered_df['total_cases'].sum()
    )

    avg_cases = round(
        filtered_df['total_cases'].mean(),
        2
    )

    max_cases = filtered_df['total_cases'].max()

    if pd.isna(max_cases):
        max_cases = 0

    max_cases = int(max_cases)

    outbreak_weeks = int(
        (filtered_df['total_cases'] > 0).sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Cases",
        total_cases
    )

    col2.metric(
        "Average Cases",
        avg_cases
    )

    col3.metric(
        "Max Cases",
        max_cases
    )

    col4.metric(
        "Outbreak Weeks",
        outbreak_weeks
    )

    if page == "Overview":

        st.subheader("📈 Weekly Disease Trend")

        fig = px.line(
            filtered_df,
            x='weekofyear',
            y='total_cases',
            markers=True,
            title='Weekly Disease Cases'
        )

        fig.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        filtered_df = filtered_df.copy()

        filtered_df['rolling_avg'] = (
            filtered_df['total_cases']
            .rolling(window=3)
            .mean()
        )

        fig2 = px.line(
            filtered_df,
            x='weekofyear',
            y='rolling_avg',
            title='3-Week Rolling Average'
        )

        fig2.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    elif page == "Prediction":

        st.subheader("🔍 Outbreak Prediction")

        if st.button("Predict Outbreak Risk"):

            latest_data = X.iloc[-1:]

            prediction = model.predict(
                latest_data
            )

            probability = (
                model.predict_proba(latest_data)[0][1]
            ) * 100

            st.progress(
                int(probability)
            )

            st.metric(
                "Prediction Confidence",
                f"{probability:.2f}%"
            )

            if prediction[0] == 1:

                st.error(
                    f"⚠️ High Outbreak Risk ({probability:.2f}%)"
                )

            else:

                st.success(
                    f"✅ Low Outbreak Risk ({probability:.2f}%)"
                )

    elif page == "Analytics":

        st.subheader("🔥 Feature Importance")

        importance = model.feature_importances_

        feature_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        })

        feature_df = feature_df.sort_values(
            by='Importance',
            ascending=False
        )

        fig3 = px.bar(
            feature_df.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Important Features'
        )

        fig3.update_layout(
            template="plotly_dark",
            height=600
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        st.subheader("📊 Dataset Preview")

        st.dataframe(
            filtered_df.head(20),
            use_container_width=True
        )

        st.subheader("📌 Statistical Summary")

        st.write(
            filtered_df.describe()
        )


if __name__ == "__main__":
    run_dashboard()

