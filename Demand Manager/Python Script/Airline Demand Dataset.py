#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Parameters
np.random.seed(42)
random.seed(42)
n_days = 180  # 6 months of data
origins = ["AUH", "LHR", "JFK", "DEL", "SYD"]  # sample airports
destinations = ["LHR", "JFK", "DEL", "SYD", "BKK"]
classes = ["Economy", "Business", "First"]
competitors = ["Emirates", "Qatar Airways", "Turkish Airlines"]

# Generate date range
start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_days)]

# Create dataset
records = []
for date in dates:
    for origin in origins:
        for dest in destinations:
            if origin == dest:
                continue
            for cls in classes:
                # Base demand depends on route popularity
                base_demand = np.random.poisson(lam=80 if cls == "Economy" else (20 if cls == "Business" else 5))

                # Add weekend/weekday effect
                if date.weekday() in [4, 5]:  # Friday, Saturday peak
                    base_demand = int(base_demand * 1.3)

                # Add random event spikes (holidays, expos, etc.)
                event_multiplier = 1
                if date in [datetime(2024, 1, 10), datetime(2024, 2, 15), datetime(2024, 3, 25)]:
                    event_multiplier = 2

                demand = int(base_demand * event_multiplier * np.random.uniform(0.8, 1.2))

                # Average ticket price (fluctuates)
                avg_price = np.random.normal(
                    loc=500 if cls == "Economy" else (2000 if cls == "Business" else 5000),
                    scale=100
                )

                # Competitor average price on same route
                competitor_prices = {
                    comp: round(avg_price * np.random.uniform(0.85, 1.15), 2)
                    for comp in competitors
                }

                # Revenue = demand * avg_price
                revenue = demand * avg_price

                records.append([
                    date, origin, dest, cls, demand, round(avg_price, 2),
                    competitor_prices, round(revenue, 2)
                ])

# Create DataFrame
df = pd.DataFrame(records, columns=[
    "date", "origin", "destination", "class", "demand", "avg_ticket_price",
    "competitor_prices", "revenue"
])

# Save dataset
df.to_csv("airline_demand_data.csv", index=False)

print("Sample dataset created with", len(df), "rows.")
print(df.head())


# In[ ]:


import pandas as pd
import mysql.connector
import json

# Load CSV
df = pd.read_csv("airline_demand_data.csv")

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="john",
    password="john",
    database="airline_demand"
)
cursor = conn.cursor()

# Insert data
for _, row in df.iterrows():
    sql = """
    INSERT INTO demand_data (date, origin, destination, class, demand, avg_ticket_price, competitor_prices, revenue)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        row['date'],
        row['origin'],
        row['destination'],
        row['class'],
        int(row['demand']),
        float(row['avg_ticket_price']),
        json.dumps(row['competitor_prices']),  # Convert dict to JSON string
        float(row['revenue'])
    )
    cursor.execute(sql, values)

conn.commit()
cursor.close()
conn.close()



# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load dataset
df = pd.read_csv("airline_demand_data.csv", parse_dates=["date"])

# =============================
# 1. Demand Trend Analysis
# =============================
demand_trend = df.groupby("date")["demand"].sum()
plt.figure(figsize=(12, 5))
plt.plot(demand_trend.index, demand_trend.values, label="Total Demand")
plt.title("Overall Passenger Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.show()

# =============================
# 2. Revenue by Route and Class
# =============================
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df.groupby(["origin", "destination", "class"])["revenue"].sum().reset_index(),
    x="class", y="revenue", hue="origin"
)
plt.title("Revenue Breakdown by Route Class and Origin")
plt.show()

# =============================
# 3. Competitor Pricing Analysis
# =============================
# Extract competitor prices from dictionary column
df_long = df.explode("competitor_prices")

# Convert competitor_prices dict to multiple rows
df_comp = df[["date", "origin", "destination", "class", "competitor_prices"]].copy()
df_comp = df_comp.explode("competitor_prices")

# Example: Focus on AUH-LHR Economy average competitor prices
route_filter = df[(df["origin"]=="AUH") & (df["destination"]=="LHR") & (df["class"]=="Economy")]
plt.figure(figsize=(12, 5))
plt.plot(route_filter["date"], route_filter["avg_ticket_price"], label="Etihad Price", color="blue")
plt.plot(route_filter["date"], [list(x.values())[0] for x in route_filter["competitor_prices"]], label="Competitor Price Sample", color="red")
plt.title("Etihad vs Competitor Pricing (AUH-LHR Economy)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# =============================
# 4. Forecasting Demand (Example AUH-LHR Economy)
# =============================
series = route_filter.set_index("date")["demand"]
model = ExponentialSmoothing(series, seasonal="add", seasonal_periods=7)
fit = model.fit()
forecast = fit.forecast(30)

plt.figure(figsize=(12, 5))
plt.plot(series.index, series.values, label="Historical Demand")
plt.plot(forecast.index, forecast.values, label="Forecasted Demand", linestyle="--")
plt.title("30-Day Demand Forecast (AUH-LHR Economy)")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.show()

# =============================
# 5. Revenue Optimization Scenario
# =============================
# Simulate effect of a 10% price increase on AUH-LHR Economy demand
price_elasticity = -0.6  # assumed elasticity
baseline = route_filter.copy()

baseline["adjusted_demand"] = baseline["demand"] * (1 + price_elasticity * 0.10)
baseline["adjusted_revenue"] = baseline["adjusted_demand"] * (baseline["avg_ticket_price"] * 1.10)

plt.figure(figsize=(12, 5))
plt.plot(baseline["date"], baseline["revenue"], label="Baseline Revenue")
plt.plot(baseline["date"], baseline["adjusted_revenue"], label="+10% Price Revenue", linestyle="--")
plt.title("Revenue Optimization Simulation (AUH-LHR Economy)")
plt.xlabel("Date")
plt.ylabel("Revenue (USD)")
plt.legend()
plt.show()


# In[ ]:




