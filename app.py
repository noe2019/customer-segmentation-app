import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data import simulate

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")
st.title("👥 Customer Segmentation – RFM + K-Means")
st.caption("Identify high-value customer segments from transaction history · simulated retail data (or upload your own)")


@st.cache_data
def sample():
    return simulate()


def rfm_table(tx):
    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    ref = tx.date.max() + pd.Timedelta(days=1)
    rfm = tx.groupby("customer_id").agg(recency=("date", lambda d: (ref - d.max()).days),
                                         frequency=("date", "count"), monetary=("amount", "sum"))
    return rfm


@st.cache_data
def cluster(rfm, k):
    Xs = StandardScaler().fit_transform(np.log1p(rfm[["recency", "frequency", "monetary"]]))
    inertia, sil = {}, {}
    for kk in range(2, 9):
        km = KMeans(kk, n_init=10, random_state=0).fit(Xs)
        inertia[kk] = km.inertia_
        sil[kk] = silhouette_score(Xs, km.labels_, sample_size=min(2000, len(Xs)), random_state=0)
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(Xs)
    return labels, inertia, sil


def name_segments(prof):
    score = (-prof.recency.rank() + prof.frequency.rank() + prof.monetary.rank())
    names = ["Champions", "Loyal customers", "Potential loyalists", "Needs attention", "At risk", "Hibernating", "Lost", "Lost (low value)"]
    order = score.sort_values(ascending=False).index
    return {c: names[i] if i < len(names) else f"Segment {i+1}" for i, c in enumerate(order)}


with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Transactions CSV (customer_id, date, amount)", type="csv")
    k = st.slider("Number of segments (k)", 2, 8, 5)
    st.markdown("[Portfolio](https://noe2019.github.io/DA-Portfolio-Website/)")

tx = pd.read_csv(up) if up else sample()
missing = {"customer_id", "date", "amount"} - set(tx.columns)
if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

rfm = rfm_table(tx)
labels, inertia, sil = cluster(rfm, k)
rfm["cluster"] = labels
prof = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
names = name_segments(prof)
rfm["segment"] = rfm.cluster.map(names)

c = st.columns(4)
c[0].metric("Customers", f"{len(rfm):,}")
c[1].metric("Transactions", f"{len(tx):,}")
c[2].metric("Revenue", f"${tx.amount.sum():,.0f}")
c[3].metric(f"Silhouette (k={k})", f"{sil[k]:.2f}")

summary = rfm.groupby("segment").agg(customers=("recency", "size"), avg_recency_days=("recency", "mean"),
                                     avg_frequency=("frequency", "mean"), avg_monetary=("monetary", "mean"),
                                     revenue=("monetary", "sum")).sort_values("revenue", ascending=False)
summary["revenue_share"] = summary.revenue / summary.revenue.sum()

t1, t2, t3 = st.tabs(["Segments", "Explore", "Choosing k"])
with t1:
    st.dataframe(summary.style.format({"avg_recency_days": "{:.0f}", "avg_frequency": "{:.1f}", "avg_monetary": "${:,.0f}",
                                       "revenue": "${:,.0f}", "revenue_share": "{:.0%}"}), use_container_width=True)
    l, r = st.columns(2)
    l.plotly_chart(px.pie(summary.reset_index(), names="segment", values="customers", title="Share of customers", hole=.4), use_container_width=True)
    r.plotly_chart(px.pie(summary.reset_index(), names="segment", values="revenue", title="Share of revenue", hole=.4), use_container_width=True)
    st.download_button("Download customer segments (CSV)", rfm.reset_index().to_csv(index=False), "segments.csv", "text/csv")
with t2:
    st.plotly_chart(px.scatter_3d(rfm.reset_index().sample(min(1500, len(rfm)), random_state=0), x="recency", y="frequency",
                                  z="monetary", color="segment", opacity=.6, log_z=True, height=600), use_container_width=True)
with t3:
    l, r = st.columns(2)
    l.plotly_chart(px.line(x=list(inertia), y=list(inertia.values()), markers=True, labels={"x": "k", "y": "Inertia"}, title="Elbow method"), use_container_width=True)
    r.plotly_chart(px.line(x=list(sil), y=list(sil.values()), markers=True, labels={"x": "k", "y": "Silhouette"}, title="Silhouette score"), use_container_width=True)
