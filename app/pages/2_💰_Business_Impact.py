import streamlit as st

st.set_page_config(page_title="Business Impact", layout="centered")

st.title("💰 Business Impact Analysis")
st.write("""
Based on model evaluation on held-out test data, here's the estimated 
business value of using this churn prediction model for targeted retention campaigns.
""")

c1, c2, c3 = st.columns(3)
c1.metric("High-Risk Customers Identified", "282")
c2.metric("Actual Churners Among Them", "185")
c3.metric("Detection Rate", "65.6%")

st.divider()

c4, c5, c6 = st.columns(3)
c4.metric("Revenue at Risk", "₹3,49,334")
c5.metric("Retention Campaign Cost", "₹56,400")
c6.metric("Net Potential Savings", "₹2,92,934", "6.2x ROI")

st.divider()
st.subheader("📌 Methodology")
st.write("""
- Top 20% highest-risk customers (by predicted churn probability) were targeted
- Assumed retention offer cost: ₹200 per customer
- Assumed average customer lifetime value: 24 months × Monthly Charges
- These are illustrative assumptions; real-world values would be validated with business stakeholders
""")

import plotly.graph_objects as go
fig = go.Figure(data=[
    go.Bar(name='Amount (₹)', x=['Revenue at Risk', 'Campaign Cost', 'Net Savings'],
           y=[349334, 56400, 292934],
           marker_color=['crimson', 'orange', 'seagreen'])
])
fig.update_layout(title="Business Impact Breakdown")
st.plotly_chart(fig, use_container_width=True)