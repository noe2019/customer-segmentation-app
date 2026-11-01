# Customer Segmentation (RFM + K-Means)

RFM analysis and K-Means clustering to identify high-value customer segments.

**Live app:** https://noe-customer-segments-15f93c700040.herokuapp.com/
**Portfolio:** https://noe2019.github.io/DA-Portfolio-Website/

## Features
- RFM computation from transactions
- Elbow and silhouette diagnostics
- Segment profiles and naming
- Upload your own transactions CSV and download segments

## Tech stack
Python, scikit-learn, Streamlit, Plotly

## Data
All data in this app is **simulated** (generated in `data.py` with a fixed random seed) to reproduce the structure of the original project without exposing real or confidential records.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed on Heroku (stack `heroku-24`, Python 3.12) using the `Procfile` in this repository.

---
Author: Noé Carème Fouotsa Manfouo
