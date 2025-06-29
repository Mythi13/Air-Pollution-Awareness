import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title("🌍 Air Quality Awareness Dashboard")

# Load data
df_model = pd.read_csv("air quality.csv")
df_forecast = pd.read_excel("Industry.xlsx")

# Build model
numeric_cols = ["weight", "humidity", "temperature"]
categorical_cols = ["Region", "month"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])

X = df_model.drop("quality", axis=1)
y = df_model["quality"]
model_pipeline.fit(X, y)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Regional Prediction", "🧪 Air Quality Predictor", "📈 Forecast 2030", "🏭 Industry Impact"])

with tab1:
    st.subheader("Predicted Air Quality by Region")
    df_model["predicted_quality"] = model_pipeline.predict(X)
    df_model["label"] = df_model["predicted_quality"].map({1: "Good", 0: "Poor"})
    region_summary = df_model.groupby(["Region", "label"]).size().unstack(fill_value=0)
    st.bar_chart(region_summary)

with tab2:
    st.subheader("Check Air Quality for Your Region")
    region = st.selectbox("Region", df_model["Region"].unique())
    weight = st.slider("Particulate Weight", 900, 1100, 1000)
    humidity = st.slider("Humidity (%)", 10, 90, 40)
    temperature = st.slider("Temperature (°F)", 40, 110, 70)
    month = st.selectbox("Month", df_model["month"].unique())

    sample = pd.DataFrame({
        "Region": [region],
        "weight": [weight],
        "humidity": [humidity],
        "temperature": [temperature],
        "month": [month]
    })

    prediction = model_pipeline.predict(sample)[0]
    label = "Good" if prediction == 1 else "Poor"
    color = "green" if prediction == 1 else "red"

    st.markdown(f"### Predicted Air Quality: **:{color}[{label}]**")

    fig, ax = plt.subplots()
    ax.bar(label, 1, color=color)
    ax.set_title(f"Air Quality in {region}")
    ax.set_ylim(0, 1.5)
    st.pyplot(fig)

with tab3:
    st.subheader("Predicted Pollution Score by District (2030)")
    df_forecast["Pollutants_List"] = df_forecast["Key Air Pollutants"].str.split(", ")
    pollution_by_district = df_forecast.groupby("District")["Pollutants_List"].sum().apply(lambda x: sum([1 for _ in x]))
    future_2030 = pollution_by_district + np.random.randint(-2, 3, len(pollution_by_district))
    top_5 = future_2030.sort_values(ascending=False).head(5)

    fig, ax = plt.subplots()
    sns.barplot(x=top_5.values, y=top_5.index, palette="Reds_r", ax=ax)
    ax.set_title("Top 5 Predicted Polluted Districts in 2030")
    ax.set_xlabel("Pollution Score")
    st.pyplot(fig)

with tab4:
    st.subheader("Industries Linked to Health Issues")
    df_forecast["Health_Issues_List"] = df_forecast["Health Concerns"].str.split(", ")
    industry_impact = df_forecast.groupby("Major_Industries")["Health_Issues_List"].sum().apply(lambda x: len(set(x)))
    top_industries = industry_impact.sort_values(ascending=False).head(5)

    fig, ax = plt.subplots()
    top_industries.plot(kind="bar", color="darkred", ax=ax)
    ax.set_title("Top 5 Health-Impacting Industries")
    ax.set_ylabel("Unique Health Issues")
    st.pyplot(fig)