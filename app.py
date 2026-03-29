import os
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "customer_data.csv"
TOP_N_COUNTRIES = 5
RANDOM_STATE = 42

NUMERIC_FEATURES = ["age", "income", "purchase_frequency"]
CATEGORICAL_FEATURES = ["gender", "education", "country"]
TARGET_COL = "spending"
ID_COL = "name"


@dataclass
class ModelArtifacts:
    linear_model: Pipeline
    rf_model: Pipeline
    metrics: pd.DataFrame
    best_model_name: str
    country_top_n: set[str]
    country_options: list[str]
    low_threshold: float
    high_threshold: float


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file '{path}' was not found. "
            "Please place your Kaggle CSV file in the project folder."
        )

    df = pd.read_csv(path)
    expected_cols = {
        ID_COL,
        "age",
        "income",
        "purchase_frequency",
        "gender",
        "education",
        "country",
        TARGET_COL,
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df.drop_duplicates().copy()
    return df


def map_top_n_country(series: pd.Series, top_n_countries: set[str]) -> pd.Series:
    return series.apply(lambda x: x if x in top_n_countries else "Other")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: KFold) -> dict[str, float]:
    preds = cross_val_predict(model, X, y, cv=cv, n_jobs=None)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


@st.cache_resource(show_spinner=True)
def train_models(data_path: str) -> ModelArtifacts:
    df = load_dataset(data_path)
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL, ID_COL]).copy()

    country_options = sorted(X["country"].astype(str).unique().tolist())

    top_n_countries = set(X["country"].value_counts().head(TOP_N_COUNTRIES).index)
    X["country"] = map_top_n_country(X["country"], top_n_countries)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    linear_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            ),
        ]
    )

    linear_grid = GridSearchCV(
        estimator=linear_pipeline,
        param_grid={"model__fit_intercept": [True, False]},
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )

    rf_grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid={
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 3],
        },
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )

    linear_grid.fit(X, y)
    rf_grid.fit(X, y)

    best_linear = linear_grid.best_estimator_
    best_rf = rf_grid.best_estimator_

    linear_metrics = evaluate_model(best_linear, X, y, cv)
    rf_metrics = evaluate_model(best_rf, X, y, cv)

    metrics = pd.DataFrame(
        {
            "Linear Regression": linear_metrics,
            "Random Forest": rf_metrics,
        }
    ).T

    best_model_name = metrics["RMSE"].idxmin()

    low_threshold = float(y.quantile(0.33))
    high_threshold = float(y.quantile(0.66))

    best_linear.fit(X, y)
    best_rf.fit(X, y)

    return ModelArtifacts(
        linear_model=best_linear,
        rf_model=best_rf,
        metrics=metrics,
        best_model_name=best_model_name,
        country_top_n=top_n_countries,
        country_options=country_options,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )


def segment_customer(predicted_spending: float, low_t: float, high_t: float) -> tuple[str, str]:
    if predicted_spending <= low_t:
        return (
            "Low Value Customer",
            "Use strong discount offers and starter bundles to improve conversion and repeat purchases.",
        )
    if predicted_spending <= high_t:
        return (
            "Medium Value Customer",
            "Use personalized bundle promotions and cross-sell campaigns to increase average order value.",
        )
    return (
        "High Value Customer",
        "Use premium product ads, loyalty perks, and exclusive early-access campaigns to maximize retention.",
    )


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _clean_feature_name(name: str) -> str:
    cleaned = name.replace("num__", "").replace("cat__", "")
    return cleaned.replace("_", " ")


def get_top_feature_importance(model_pipeline: Pipeline, top_n: int = 8) -> pd.DataFrame:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    estimator = model_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        raw_scores = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        raw_scores = np.abs(np.asarray(estimator.coef_, dtype=float).ravel())
    else:
        return pd.DataFrame(columns=["feature", "importance", "importance_pct"])

    scores = np.maximum(raw_scores, 0)
    total = float(scores.sum())
    if total == 0:
        normalized = scores
    else:
        normalized = (scores / total) * 100

    importance_df = pd.DataFrame(
        {
            "feature": [_clean_feature_name(n) for n in feature_names],
            "importance": scores,
            "importance_pct": normalized,
        }
    )
    return importance_df.sort_values("importance", ascending=False).head(top_n)


def render_importance_chart(importance_df: pd.DataFrame, title: str, color: str) -> None:
    chart = (
        alt.Chart(importance_df)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("importance_pct:Q", title="Importance (%)"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.value(color),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance_pct:Q", title="Importance (%)", format=".2f"),
            ],
        )
        .properties(height=280, title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def render_header() -> None:
    st.set_page_config(
        page_title="Personalized Marketing & Advertising System",
        page_icon="📊",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        :root {
            --brand-900: #0b2f4a;
            --brand-700: #0f5fa8;
            --brand-500: #2f8bcf;
            --brand-200: #d6e9f8;
            --text-900: #1f2b37;
            --text-700: #334155;
            --ok-700: #166534;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 10%, #b9dcff 0%, rgba(185, 220, 255, 0.92) 24%, rgba(185, 220, 255, 0) 56%),
                radial-gradient(circle at 92% 12%, #c7e5ff 0%, rgba(199, 229, 255, 0.9) 22%, rgba(199, 229, 255, 0) 52%),
                linear-gradient(180deg, #e6f3ff 0%, #d6ebff 52%, #eef7ff 100%);
            color-scheme: light;
        }

        .stMarkdown, .stText, .stCaption, p, h1, h2, h3, h4 {
            color: var(--text-900) !important;
        }

        div[data-testid="stForm"] {
            border: 1px solid #dbe7f3;
            border-radius: 14px;
            padding: 0.75rem 0.85rem 0.8rem 0.85rem;
            background: linear-gradient(180deg, #ffffff 0%, #f8fcff 100%);
            box-shadow: 0 8px 24px rgba(19, 79, 132, 0.08);
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] div[data-baseweb="input"] > div {
            background: #ffffff !important;
            border: 1px solid #c7d8e8 !important;
            border-radius: 10px !important;
            box-shadow: 0 3px 10px rgba(15, 95, 168, 0.08) !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="input"] span {
            color: #16324c !important;
        }

        div[data-baseweb="select"] div[role="combobox"] {
            color: #16324c !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] div[role="combobox"] input {
            color: #16324c !important;
            -webkit-text-fill-color: #16324c !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] svg {
            color: #16324c !important;
            fill: #16324c !important;
            opacity: 1 !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            color: #16324c !important;
            opacity: 1 !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
            color: #16324c !important;
            -webkit-text-fill-color: #16324c !important;
            opacity: 1 !important;
        }

        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: #0f5fa8 !important;
            border: 2px solid #ffffff !important;
        }

        .stSlider [data-baseweb="slider"] > div > div {
            background-color: #b7d8f3 !important;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, #0d4c80 0%, #0f5fa8 45%, #2f8bcf 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 0.55rem 0.9rem !important;
            box-shadow: 0 10px 22px rgba(15, 95, 168, 0.25) !important;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-1px);
            filter: brightness(1.04);
        }

        div[data-testid="stForm"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stTextInput"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label {
            color: #243746 !important;
            font-weight: 600 !important;
        }

        .stDataFrame {
            border: 1px solid #dbe7f3;
            border-radius: 12px;
            overflow: hidden;
        }

        .hero {
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            background: linear-gradient(120deg, #08395f 0%, #0f5fa8 42%, #2f8bcf 74%, #6bc4dd 100%);
            color: white;
            margin-bottom: 1.0rem;
            box-shadow: 0 12px 30px rgba(19, 79, 132, 0.25);
        }

        .hero-icon-row {
            display: flex;
            gap: 0.6rem;
            margin-top: 0.75rem;
            flex-wrap: wrap;
        }

        .hero-pill {
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .card {
            border: 1px solid #dbe7f3;
            border-radius: 14px;
            background: linear-gradient(180deg, #ffffff 0%, #f9fcff 100%);
            padding: 0.95rem 1.05rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        }
        .section-title {
            font-weight: 800;
            color: var(--brand-900);
            margin-bottom: 0.5rem;
            letter-spacing: 0.2px;
        }
        .chip-wrap {
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
            margin-top: 0.45rem;
        }
        .chip {
            display: inline-block;
            background: #eaf3fb;
            color: #16476d;
            border: 1px solid #c8def0;
            border-radius: 999px;
            padding: 0.24rem 0.62rem;
            font-size: 0.81rem;
            font-weight: 600;
        }
        .value {
            font-size: 1.55rem;
            font-weight: 700;
            color: #0f5fa8;
        }
        .small-label {
            color: #44556a;
            font-size: 0.9rem;
            margin-top: 0.2rem;
        }
        .good {
            color: var(--ok-700);
            font-weight: 700;
        }
        .segment {
            font-size: 1.15rem;
            font-weight: 700;
            color: #0d3b66;
        }
        .rec {
            color: #17324d;
            font-size: 0.98rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h2 style="margin:0;">Personalized Marketing & Advertising System</h2>
            <p style="margin:0.35rem 0 0 0;opacity:0.95;">
                Predict customer annual spending using Linear Regression and Random Forest,
                compare models, segment customers, and generate targeted campaign recommendations.
            </p>
            <div class="hero-icon-row">
                <span class="hero-pill">📈 Spending Prediction</span>
                <span class="hero-pill">🧠 Dual-Model Comparison</span>
                <span class="hero-pill">🎯 Customer Segmentation</span>
                <span class="hero-pill">📣 Marketing Strategy</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    render_header()

    with st.spinner("Training models and preparing dashboard..."):
        artifacts = train_models(DATA_PATH)

    left, right = st.columns([1.05, 1.35], gap="large")

    with left:
        st.markdown("<div class='section-title'>🧾 Customer Input</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card">
                <div class="small-label"><b>Required Inputs Included:</b></div>
                <div class="chip-wrap">
                    <span class="chip">Age</span>
                    <span class="chip">Gender</span>
                    <span class="chip">Education</span>
                    <span class="chip">Income</span>
                    <span class="chip">Country</span>
                    <span class="chip">Purchase Frequency</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("customer_form"):
            age = st.slider("Age", min_value=18, max_value=80, value=35)
            gender = st.selectbox("Gender", options=["Male", "Female"])
            education = st.selectbox(
                "Education",
                options=["High School", "Bachelor", "Master", "PhD"],
            )
            income = st.number_input(
                "Annual Income",
                min_value=1,
                max_value=500000,
                value=75000,
                step=1,
            )
            country_input = st.selectbox(
                "Country",
                options=artifacts.country_options + ["Other"],
                index=(artifacts.country_options.index("USA") if "USA" in artifacts.country_options else 0),
                help="Select customer country from dataset options.",
            )
            purchase_frequency = st.slider(
                "Purchase Frequency (per period)",
                min_value=1,
                max_value=40,
                value=12,
            )
            submitted = st.form_submit_button("Generate Prediction", use_container_width=True)

    with right:
        st.markdown("<div class='section-title'>📊 Model Quality Comparison (5-Fold CV)</div>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        metrics_display = artifacts.metrics.copy()
        metrics_display["RMSE"] = metrics_display["RMSE"].map(lambda x: f"{x:,.2f}")
        metrics_display["MAE"] = metrics_display["MAE"].map(lambda x: f"{x:,.2f}")
        metrics_display["R2"] = metrics_display["R2"].map(lambda x: f"{x:.3f}")
        st.dataframe(metrics_display, use_container_width=True)

        st.markdown(
            f"Best-performing model by RMSE: "
            f"<span class='good'>{artifacts.best_model_name}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>🔍 Feature Importance Insights</div>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        imp_col1, imp_col2 = st.columns(2, gap="medium")

        linear_importance = get_top_feature_importance(artifacts.linear_model, top_n=8)
        rf_importance = get_top_feature_importance(artifacts.rf_model, top_n=8)

        with imp_col1:
            st.caption("Linear Regression (absolute coefficients)")
            render_importance_chart(linear_importance, "Top Linear Drivers", "#2f8bcf")

        with imp_col2:
            st.caption("Random Forest (feature importance)")
            render_importance_chart(rf_importance, "Top RF Drivers", "#0f5fa8")

        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        country = country_input
        country_grouped = country if country in artifacts.country_top_n else "Other"

        customer_df = pd.DataFrame(
            [
                {
                    "age": age,
                    "income": float(income),
                    "purchase_frequency": purchase_frequency,
                    "gender": gender,
                    "education": education,
                    "country": country_grouped,
                }
            ]
        )

        pred_linear = float(artifacts.linear_model.predict(customer_df)[0])
        pred_rf = float(artifacts.rf_model.predict(customer_df)[0])

        better_model = (
            "Linear Regression"
            if artifacts.metrics.loc["Linear Regression", "RMSE"]
            <= artifacts.metrics.loc["Random Forest", "RMSE"]
            else "Random Forest"
        )

        chosen_prediction = pred_linear if better_model == "Linear Regression" else pred_rf
        segment, recommendation = segment_customer(
            chosen_prediction,
            artifacts.low_threshold,
            artifacts.high_threshold,
        )

        st.markdown("<div class='section-title'>💡 Prediction Dashboard</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div class="card">
                    <div class="small-label">Linear Regression Prediction</div>
                    <div class="value">{format_currency(pred_linear)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="card">
                    <div class="small-label">Random Forest Prediction</div>
                    <div class="value">{format_currency(pred_rf)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
                <div class="card">
                    <div class="small-label">Selected Model</div>
                    <div class="value" style="font-size:1.2rem;">{better_model}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div class='section-title'>🧭 Customer Segment & Personalized Strategy</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card">
                <div class="segment">{segment}</div>
                <div class="small-label">Predicted Spending (using selected model)</div>
                <div class="value">{format_currency(chosen_prediction)}</div>
                <p class="rec"><b>Recommended Advertising Strategy:</b> {recommendation}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-title'>👤 Entered Customer Profile</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card">
                <div class="small-label"><b>Age:</b> {age}</div>
                <div class="small-label"><b>Gender:</b> {gender}</div>
                <div class="small-label"><b>Education:</b> {education}</div>
                <div class="small-label"><b>Annual Income:</b> {format_currency(float(income))}</div>
                <div class="small-label"><b>Country (selected):</b> {country}</div>
                <div class="small-label"><b>Country (mapped for model):</b> {country_grouped}</div>
                <div class="small-label"><b>Purchase Frequency:</b> {purchase_frequency}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
