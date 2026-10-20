
from io import BytesIO

import boto3
import pandas as pd
import plotly.express as px
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CloudSentinel AI",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONFIGURATION
# ============================================================

S3_BUCKET = "cloudsentinel-ai-102026"
S3_REGION = "ap-south-1"
S3_PREFIX = "predictions/"

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1220;
        color: #e5edf8;
    }

    [data-testid="stMetric"] {
        background: #141f32;
        border: 1px solid #263750;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #a8bad2;
    }

    div[data-testid="stMetricValue"] {
        color: #f1f5f9;
    }

    h1, h2, h3 {
        color: #e5edf8;
    }

    section[data-testid="stSidebar"] {
        background-color: #101a2b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# S3 HELPERS
# ============================================================

@st.cache_resource
def get_s3_client(region):
    """Reuse the local AWS SDK client and its configured credentials."""
    return boto3.client("s3", region_name=region)


def find_latest_prediction_csv(s3_client):
    """Find the newest CSV under predictions/, handling pagination."""
    paginator = s3_client.get_paginator("list_objects_v2")
    latest = None

    for page in paginator.paginate(
        Bucket=S3_BUCKET,
        Prefix=S3_PREFIX,
    ):
        for obj in page.get("Contents", []):
            if not obj["Key"].lower().endswith(".csv"):
                continue

            if latest is None or obj["LastModified"] > latest["LastModified"]:
                latest = obj

    return latest


@st.cache_data
def load_prediction_csv(bucket, key, region, last_modified):
    """Download and parse the selected S3 prediction CSV."""
    client = boto3.client("s3", region_name=region)

    response = client.get_object(
        Bucket=bucket,
        Key=key,
    )

    content = response["Body"].read()

    return pd.read_csv(BytesIO(content))


def clear_dashboard_cache():
    """Force the next rerun to fetch and parse fresh data."""
    load_prediction_csv.clear()


# ============================================================
# HEADER
# ============================================================

st.title("☁️ CloudSentinel AI")

st.markdown(
    """
    **Intelligent Cloud Resource Monitoring & Cost Anomaly Detection**

    Monitor synthetic cloud resource metrics, inspect anomaly predictions,
    and explore the output produced by your deployed AWS Lambda model.
    """
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Dashboard Controls")

    st.caption("Model: Isolation Forest V3")
    st.caption("Inference: AWS Lambda")
    st.caption(f"Region: {S3_REGION}")
    st.caption("Storage: Private Amazon S3")

    if st.button("🔄 Refresh from S3", use_container_width=True):
        clear_dashboard_cache()
        st.rerun()

    st.divider()

    st.markdown("**Data source**")
    st.code(
        f"s3://{S3_BUCKET}/{S3_PREFIX}",
        language="text",
    )

# ============================================================
# LOAD LATEST S3 DATA
# ============================================================

try:
    s3 = get_s3_client(S3_REGION)
    latest_object = find_latest_prediction_csv(s3)

    if latest_object is None:
        st.warning(
            "No prediction CSV was found in the S3 predictions/ prefix. "
            "Invoke your Lambda function to generate an output file first."
        )
        st.stop()

    latest_key = latest_object["Key"]
    latest_modified = latest_object["LastModified"].isoformat()

    df = load_prediction_csv(
        S3_BUCKET,
        latest_key,
        S3_REGION,
        latest_modified,
    )

except (BotoCoreError, ClientError) as exc:
    st.error("Unable to load prediction data from private S3.")
    st.write(
        "Check your local AWS CLI credentials, IAM permissions, "
        "bucket name, and AWS region."
    )
    st.code(str(exc))
    st.stop()

except (ValueError, pd.errors.ParserError) as exc:
    st.error("The prediction CSV could not be parsed.")
    st.code(str(exc))
    st.stop()

if df.empty:
    st.warning("The latest prediction CSV contains no rows.")
    st.stop()

# ============================================================
# VALIDATE AND PREPARE DATA
# ============================================================

required_columns = {
    "resource_id",
    "prediction_label",
    "anomaly_score",
    "cpu_utilization",
    "memory_utilization",
    "estimated_cost",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(
        "The CSV is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

df["resource_id"] = df["resource_id"].astype(str)
df["prediction_label"] = (
    df["prediction_label"].astype(str).str.strip().str.lower()
)
df["anomaly_score"] = pd.to_numeric(
    df["anomaly_score"], errors="coerce"
)

for column in [
    "cpu_utilization",
    "memory_utilization",
    "estimated_cost",
    "is_anomaly",
    "is_anomaly_prediction",
]:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

st.caption(f"**Loaded from S3:** `s3://{S3_BUCKET}/{latest_key}`")
st.caption(f"**Last modified:** {latest_modified}")

# ============================================================
# OVERVIEW METRICS
# ============================================================

total_records = len(df)
predicted_anomalies = int(
    (df["prediction_label"] == "anomaly").sum()
)
predicted_normal = total_records - predicted_anomalies

anomaly_rate = (
    predicted_anomalies / total_records * 100
    if total_records
    else 0
)

st.subheader("📊 System Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Records Analyzed", f"{total_records:,}")
c2.metric("Predicted Anomalies", f"{predicted_anomalies:,}")
c3.metric("Predicted Normal", f"{predicted_normal:,}")
c4.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

# ============================================================
# MODEL EVALUATION
# ============================================================

st.subheader("🎯 Model Evaluation")

if "is_anomaly" in df.columns:
    evaluation = df[
        df["is_anomaly"].isin([0, 1])
        & df["prediction_label"].isin(["normal", "anomaly"])
    ].copy()

    actual = evaluation["is_anomaly"].astype(int)
    predicted = (
        evaluation["prediction_label"] == "anomaly"
    ).astype(int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    e1, e2, e3, e4 = st.columns(4)

    e1.metric("Precision", f"{precision:.1%}")
    e2.metric("Recall", f"{recall:.1%}")
    e3.metric("F1 Score", f"{f1_score:.1%}")
    e4.metric("True Positives", str(tp))

    with st.expander("View confusion matrix"):
        confusion = pd.DataFrame(
            [
                [tn, fp],
                [fn, tp],
            ],
            index=["Actual Normal", "Actual Anomaly"],
            columns=["Predicted Normal", "Predicted Anomaly"],
        )
        st.dataframe(confusion, use_container_width=True)

    st.caption(
        "Metrics are calculated against the synthetic ground-truth "
        "labels in this CSV. They describe this test sample, not "
        "real-world AWS incidents."
    )
else:
    st.info(
        "Ground-truth labels are unavailable. Model evaluation metrics "
        "cannot be calculated for this file."
    )

# ============================================================
# FILTERS
# ============================================================

st.subheader("🔎 Explore Predictions")

f1, f2, f3 = st.columns(3)

resource_options = sorted(df["resource_id"].dropna().unique())
label_options = sorted(df["prediction_label"].dropna().unique())

with f1:
    selected_resources = st.multiselect(
        "Resource",
        resource_options,
        default=resource_options,
    )

with f2:
    selected_labels = st.multiselect(
        "Prediction",
        label_options,
        default=label_options,
    )

with f3:
    if "anomaly_type" in df.columns:
        type_options = sorted(
            df["anomaly_type"].dropna().astype(str).unique()
        )
        selected_types = st.multiselect(
            "Ground-truth anomaly type",
            type_options,
            default=type_options,
        )
    else:
        selected_types = []

filtered = df[
    df["resource_id"].isin(selected_resources)
    & df["prediction_label"].isin(selected_labels)
].copy()

if "anomaly_type" in filtered.columns:
    filtered = filtered[
        filtered["anomaly_type"].astype(str).isin(selected_types)
    ]

st.caption(f"Displaying {len(filtered):,} of {total_records:,} records")

# ============================================================
# CHART 1: PREDICTIONS BY RESOURCE
# ============================================================

st.subheader("📈 Anomaly Analytics")

left, right = st.columns(2)

with left:
    st.markdown("#### Predictions by Resource")

    if not filtered.empty:
        resource_counts = (
            filtered.groupby(
                ["resource_id", "prediction_label"]
            )
            .size()
            .reset_index(name="records")
        )

        fig = px.bar(
            resource_counts,
            x="resource_id",
            y="records",
            color="prediction_label",
            barmode="stack",
            color_discrete_map={
                "normal": "#38bdf8",
                "anomaly": "#fb7185",
            },
            labels={
                "resource_id": "Resource",
                "records": "Record count",
                "prediction_label": "Prediction",
            },
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No records match the selected filters.")

# ============================================================
# CHART 2: ANOMALY SCORE DISTRIBUTION
# ============================================================

with right:
    st.markdown("#### Anomaly Score Distribution")

    scores = filtered.dropna(subset=["anomaly_score"])

    if not scores.empty:
        fig = px.histogram(
            scores,
            x="anomaly_score",
            color="prediction_label",
            nbins=25,
            marginal="box",
            color_discrete_map={
                "normal": "#38bdf8",
                "anomaly": "#fb7185",
            },
            labels={
                "anomaly_score": "Anomaly score",
                "prediction_label": "Prediction",
            },
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No anomaly scores are available.")

# ============================================================
# CHART 3: CPU VS MEMORY
# ============================================================

left, right = st.columns(2)

with left:
    st.markdown("#### CPU vs Memory Utilization")

    if not filtered.empty:
        fig = px.scatter(
            filtered,
            x="cpu_utilization",
            y="memory_utilization",
            color="prediction_label",
            hover_data=[
                col
                for col in [
                    "resource_id",
                    "resource_type",
                    "anomaly_score",
                ]
                if col in filtered.columns
            ],
            color_discrete_map={
                "normal": "#38bdf8",
                "anomaly": "#fb7185",
            },
            labels={
                "cpu_utilization": "CPU utilization",
                "memory_utilization": "Memory utilization",
                "prediction_label": "Prediction",
            },
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for this chart.")

# ============================================================
# CHART 4: ESTIMATED COST BY RESOURCE TYPE
# ============================================================

with right:
    st.markdown("#### Estimated Cost by Resource Type")

    if "resource_type" in filtered.columns and not filtered.empty:
        cost_data = (
            filtered.groupby("resource_type", as_index=False)
            ["estimated_cost"]
            .mean()
            .sort_values("estimated_cost", ascending=False)
        )

        fig = px.bar(
            cost_data,
            x="resource_type",
            y="estimated_cost",
            labels={
                "resource_type": "Resource type",
                "estimated_cost": "Mean estimated cost",
            },
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Resource type or cost data is unavailable.")

# ============================================================
# PREDICTION TABLE
# ============================================================

st.subheader("📋 Prediction Details")

preferred_columns = [
    "timestamp",
    "resource_id",
    "resource_type",
    "cpu_utilization",
    "memory_utilization",
    "network_in",
    "request_count",
    "estimated_cost",
    "is_anomaly",
    "anomaly_type",
    "is_anomaly_prediction",
    "prediction_label",
    "anomaly_score",
]

display_columns = [
    column
    for column in preferred_columns
    if column in filtered.columns
]

st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# CSV DOWNLOAD
# ============================================================

st.download_button(
    label="⬇️ Download Filtered Predictions",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="cloudsentinel_filtered_predictions.csv",
    mime="text/csv",
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CloudSentinel AI | Isolation Forest V3 | AWS Lambda + Amazon S3"
)
st.caption(
    "Prototype note: resource metrics and estimated costs are synthetic. "
    "Predictions are model outputs, not confirmed production incidents "
    "or actual AWS billing data."
)
