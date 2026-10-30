"""Streamlit dashboard for the fleet platform.

Run with::

    streamlit run src/fleet/dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fleet import config  # noqa: E402
from fleet.db import SessionLocal, init_db  # noqa: E402
from fleet.ml.service import (  # noqa: E402
    optimize_route,
    predict_delay,
    predict_fuel_anomaly,
)
from fleet.models import Anomaly, Trip  # noqa: E402
from fleet.seed import seed_database  # noqa: E402

st.set_page_config(
    page_title="Fleet Command Centre",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATUS_COLORS = {"On_Time": "#2E8B57", "Delayed": "#E6A817", "Failed": "#C0392B"}


@st.cache_resource(show_spinner=False)
def bootstrap() -> bool:
    init_db()
    with SessionLocal() as session:
        seed_database(session)
    return True


@st.cache_data(show_spinner="Loading fleet data…")
def load_trips() -> pd.DataFrame:
    with SessionLocal() as session:
        trips = session.scalars(select(Trip)).all()
        anomaly_map = {
            anomaly.trip_id: anomaly.anomaly_type
            for anomaly in session.scalars(select(Anomaly)).all()
        }
    rows = [
        {
            "trip_id": trip.trip_id,
            "vehicle_id": trip.vehicle_id,
            "driver_id": trip.driver_id,
            "origin": trip.origin,
            "destination": trip.destination,
            "distance_km": trip.distance_km,
            "fuel_L": trip.fuel_L,
            "delivery_status": trip.delivery_status,
            "driver_score": trip.driver_score,
            "trip_start_time": trip.trip_start_time,
            "vehicle_class": trip.vehicle_class,
            "load_tonnes": trip.load_tonnes,
            "data_source": trip.data_source,
            "month": trip.month,
            "day_of_week": trip.day_of_week,
            "kmpl": trip.kmpl,
        }
        for trip in trips
    ]
    frame = pd.DataFrame(rows)
    frame["is_anomaly"] = frame["trip_id"].isin(anomaly_map)
    frame["anomaly_type"] = frame["trip_id"].map(anomaly_map).fillna("")
    frame["route"] = frame["origin"] + " → " + frame["destination"]
    frame["month_name"] = frame["trip_start_time"].dt.strftime("%b")
    return frame


@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    path = config.REPORTS / "driver_scores.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = config.REPORTS / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def kpi_row(frame: pd.DataFrame) -> None:
    clean = frame[~frame["is_anomaly"]]
    columns = st.columns(5)
    columns[0].metric("Trips", f"{len(frame):,}")
    columns[1].metric("On-time", f"{(frame.delivery_status == 'On_Time').mean():.1%}")
    columns[2].metric("Median efficiency", f"{clean['kmpl'].median():.2f} km/L")
    columns[3].metric("Anomalies", f"{int(frame['is_anomaly'].sum()):,}")
    columns[4].metric("Fleet", f"{frame.vehicle_id.nunique()} vehicles")


def page_overview(frame: pd.DataFrame) -> None:
    st.title("Fleet Command Centre")
    st.caption("Operational overview · 5,000 trips · 100 vehicles · 50 drivers · Jan–Dec 2025")
    kpi_row(frame)

    left, right = st.columns((2, 1))
    monthly = (
        frame.groupby("month")
        .agg(
            trips=("trip_id", "count"),
            on_time=("delivery_status", lambda series: (series == "On_Time").mean()),
        )
        .reset_index()
    )
    with left:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(
            x=monthly["month"], y=monthly["trips"], name="Trips", marker_color="#4C78A8"
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly["on_time"] * 100,
                name="On-time %",
                mode="lines+markers",
                line=dict(color="#2E8B57", width=3),
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="Monthly volume and on-time performance",
            xaxis_title="Month",
            yaxis_title="Trips",
            legend=dict(orientation="h", y=1.02),
            height=420,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        fig.update_yaxes(title_text="On-time %", range=[60, 100], secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        status = frame.delivery_status.value_counts().reindex(
            ["On_Time", "Delayed", "Failed"]
        )
        fig = px.pie(
            values=status.values,
            names=status.index,
            color=status.index,
            color_discrete_map=STATUS_COLORS,
            hole=0.55,
            title="Delivery status mix",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top routes by volume")
    routes = (
        frame.groupby("route")
        .agg(
            trips=("trip_id", "count"),
            on_time=("delivery_status", lambda series: (series == "On_Time").mean()),
            avg_km=("distance_km", "mean"),
        )
        .sort_values("trips", ascending=False)
        .head(10)
        .reset_index()
    )
    fig = px.bar(
        routes,
        x="trips",
        y="route",
        orientation="h",
        color="on_time",
        color_continuous_scale="RdYlGn",
        text="trips",
        labels={"trips": "Trips", "route": "", "on_time": "On-time"},
        title="Top 10 origin–destination routes",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=420,
        coloraxis_colorbar_title="On-time",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def page_delivery(frame: pd.DataFrame) -> None:
    st.title("Delivery Performance")
    kpi_row(frame)
    counts = frame.delivery_status.value_counts()
    columns = st.columns(4)
    columns[0].metric("On-time", f"{counts.get('On_Time', 0):,}")
    columns[1].metric("Delayed", f"{counts.get('Delayed', 0):,}")
    columns[2].metric("Failed", f"{counts.get('Failed', 0):,}")
    columns[3].metric("Avg distance", f"{frame.distance_km.mean():,.0f} km")

    pivot = frame.pivot_table(
        index="origin",
        columns="destination",
        values="delivery_status",
        aggfunc=lambda series: (series == "On_Time").mean(),
    ).reindex(index=config.CITIES, columns=config.CITIES)
    z = pivot.to_numpy() * 100
    text = np.where(
        np.isnan(z), "", np.char.add(np.round(np.nan_to_num(z), 1).astype(str), "%")
    )
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=config.CITIES,
            y=config.CITIES,
            colorscale="RdYlGn",
            zmin=60,
            zmax=100,
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="On-time %"),
        )
    )
    fig.update_layout(
        title="On-time rate by origin–destination pair",
        xaxis_title="Destination",
        yaxis_title="Origin",
        height=500,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Delay risk predictor")
    st.caption(
        "Decision support for proactive customer communication. "
        "The score is not used to penalise drivers."
    )
    with st.form("delay_form"):
        c1, c2, c3 = st.columns(3)
        origin = c1.selectbox("Origin", config.CITIES, index=0)
        destination = c2.selectbox("Destination", config.CITIES, index=1)
        month = c3.selectbox(
            "Month", list(range(1, 13)), index=6, format_func=lambda value: f"{value:02d}"
        )
        c4, c5, c6 = st.columns(3)
        distance = c4.slider("Distance (km)", 50, 2300, 1400, step=10)
        driver_score = c5.slider("Driver score", 60, 100, 80)
        vehicle_class = c6.selectbox("Vehicle class", list(config.VEHICLE_CLASSES))
        load = st.slider("Load (tonnes)", 0.0, 30.0, 12.0, step=0.5)
        submitted = st.form_submit_button("Predict on-time probability")

    if submitted:
        result = predict_delay(
            {
                "origin": origin,
                "destination": destination,
                "distance_km": distance,
                "driver_score": driver_score,
                "month": month,
                "vehicle_class": vehicle_class,
                "load_tonnes": load,
                "day_of_week": "Monday",
            }
        )
        probability = result["on_time_probability"]
        c1, c2, c3 = st.columns(3)
        c1.metric("On-time probability", f"{probability:.1%}")
        c2.metric("Risk level", result["risk_level"])
        c3.metric("Model", result["model"])
        st.progress(min(max(probability, 0.0), 1.0))
        st.info(
            f"Prediction: **{result['prediction']}**. "
            "Use this to prioritise support and communication, not as an automated decision."
        )


def page_fuel(frame: pd.DataFrame) -> None:
    st.title("Fuel & Anomalies")
    clean = frame[~frame["is_anomaly"]]
    columns = st.columns(4)
    columns[0].metric("Median efficiency", f"{clean['kmpl'].median():.2f} km/L")
    columns[1].metric("Total fuel", f"{frame.fuel_L.sum():,.0f} L")
    columns[2].metric("Anomaly rate", f"{frame['is_anomaly'].mean():.2%}")
    columns[3].metric("Anomalies", f"{int(frame['is_anomaly'].sum()):,}")

    left, right = st.columns(2)
    with left:
        fig = px.box(
            clean,
            x="vehicle_class",
            y="kmpl",
            color="vehicle_class",
            points="outliers",
            labels={"kmpl": "km per litre", "vehicle_class": ""},
            title="Fuel efficiency by vehicle class",
        )
        fig.update_layout(showlegend=False, height=420, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(
            frame,
            x="distance_km",
            y="fuel_L",
            color="is_anomaly",
            color_discrete_map={False: "#4C78A8", True: "#C0392B"},
            opacity=0.55,
            hover_data=["trip_id", "vehicle_class", "kmpl"],
            labels={
                "distance_km": "Distance (km)",
                "fuel_L": "Fuel (L)",
                "is_anomaly": "Anomaly",
            },
            title="Distance vs fuel consumption",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fuel anomaly checker")
    with st.form("fuel_form"):
        c1, c2, c3 = st.columns(3)
        distance = c1.number_input("Distance (km)", min_value=1.0, value=1000.0, step=10.0)
        fuel = c2.number_input("Fuel used (L)", min_value=1.0, value=280.0, step=5.0)
        vehicle_class = c3.selectbox(
            "Vehicle class", list(config.VEHICLE_CLASSES), key="fuel_class"
        )
        c4, c5 = st.columns(2)
        load = c4.number_input("Load (tonnes)", min_value=0.0, value=12.0, step=0.5)
        driver_score = c5.number_input(
            "Driver score", min_value=60, max_value=100, value=80
        )
        submitted = st.form_submit_button("Check fuel reading")

    if submitted:
        result = predict_fuel_anomaly(
            {
                "distance_km": distance,
                "fuel_L": fuel,
                "vehicle_class": vehicle_class,
                "load_tonnes": load,
                "driver_score": driver_score,
            }
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anomaly", "Yes" if result["is_anomaly"] else "No")
        c2.metric("Actual", f"{result['kmpl']:.2f} km/L")
        c3.metric("Expected", f"{result['expected_kmpl']:.2f} km/L")
        c4.metric("Deviation", f"{result['deviation_pct']:.1f}%")
        if result["is_anomaly"]:
            st.error(
                f"Flagged by **{result['method']}** (z = {result.get('z_score', 0):.2f}). "
                "Route to maintenance/investigation; do not treat as proof of misconduct."
            )
        else:
            st.success(f"Within the expected range for {vehicle_class}.")

    st.subheader("Recent flagged readings")
    flagged = frame[frame["is_anomaly"]][
        [
            "trip_id",
            "anomaly_type",
            "vehicle_id",
            "origin",
            "destination",
            "distance_km",
            "fuel_L",
            "kmpl",
        ]
    ].head(20)
    st.dataframe(flagged, use_container_width=True, hide_index=True)


def page_drivers(frame: pd.DataFrame, scores: pd.DataFrame) -> None:
    st.title("Driver Behaviour")
    metrics = load_metrics()
    fairness = metrics.get("driver_score", {}).get("fairness", {})
    if scores.empty:
        st.warning("Run `python scripts/train_models.py` to generate driver scores.")
        return

    columns = st.columns(4)
    columns[0].metric("Drivers scored", f"{len(scores)}")
    columns[1].metric("Mean score", f"{scores.score.mean():.1f}")
    columns[2].metric("Top score", f"{scores.score.max():.1f}")
    columns[3].metric(
        "Depot mean gap",
        f"{fairness.get('max_depot_mean_gap', 0):.1f} pts",
        help="Difference between the highest and lowest depot mean. Flag threshold: 10 points.",
    )

    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            scores,
            x="score",
            nbins=20,
            title="Driver score distribution",
            labels={"score": "Composite score"},
            color_discrete_sequence=["#4C78A8"],
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(
            scores.head(12).sort_values("score"),
            x="score",
            y="driver_id",
            orientation="h",
            color="score",
            color_continuous_scale="Blues",
            title="Top 12 drivers",
            labels={"score": "Composite score", "driver_id": ""},
        )
        fig.update_layout(
            height=380,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Driver detail")
    selected = st.selectbox("Select a driver", scores.driver_id.tolist())
    row = scores[scores.driver_id == selected].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{row.score:.1f}")
    c2.metric("Grade", row.grade)
    c3.metric("Trips", f"{int(row.trips)}")
    c4.metric("On-time", f"{row.on_time_rate:.1%}")

    components = pd.DataFrame(
        {
            "component": [
                "On-time (40%)",
                "Fuel efficiency (25%)",
                "Incident-free (20%)",
                "Consistency (15%)",
            ],
            "value": [
                row.on_time_component,
                row.fuel_component,
                row.incident_component,
                row.consistency_component,
            ],
        }
    )
    fig = px.bar(
        components,
        x="value",
        y="component",
        orientation="h",
        range_x=[0, 1],
        text=components["value"].map(lambda value: f"{value:.0%}"),
        title=f"Why {selected} scored {row.score:.1f}",
        labels={"value": "Component score (0–1)", "component": ""},
        color="value",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(
        height=360,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Transparent weighted formula: 60 + 40 × (0.40·on-time + 0.25·fuel + "
        "0.20·incident-free + 0.15·consistency). "
        "Used for coaching and recognition only; human review and an appeals path are mandatory."
    )

    st.subheader("Fairness audit")
    if fairness.get("by_depot"):
        depot = pd.DataFrame(fairness["by_depot"])
        st.dataframe(depot, use_container_width=True, hide_index=True)
        st.caption(
            "No protected attributes are used. Depot and experience are monitored for "
            "unintended disparities before any action."
        )


def page_routes() -> None:
    st.title("Route Optimiser")
    st.caption(
        "Exact TSP for up to 7 stops, nearest-neighbour + 2-opt for larger routes. "
        "Distances use the six-city road matrix."
    )
    with st.form("route_form"):
        origin = st.selectbox("Depot / origin", config.CITIES, index=0)
        stops = st.multiselect(
            "Stops",
            [city for city in config.CITIES if city != origin],
            default=[city for city in config.CITIES if city != origin][:4],
        )
        return_to_origin = st.checkbox("Return to origin", value=True)
        submitted = st.form_submit_button("Optimise route")

    if not submitted:
        st.info("Choose an origin and at least one stop, then optimise.")
        return
    if not stops:
        st.warning("Select at least one stop.")
        return

    result = optimize_route(origin, stops, return_to_origin)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline", f"{result['baseline_distance_km']:,.0f} km")
    c2.metric("Optimised", f"{result['optimized_distance_km']:,.0f} km")
    c3.metric("Saved", f"{result['savings_km']:,.0f} km")
    c4.metric("Saving", f"{result['savings_pct']:.1f}%")

    st.markdown(
        "**Optimised order:** "
        + " → ".join([origin, *result["optimized_order"]])
        + (" → " + origin if return_to_origin else "")
    )
    st.dataframe(pd.DataFrame(result["legs"]), use_container_width=True, hide_index=True)

    route_cities = [origin, *result["optimized_order"]]
    if return_to_origin:
        route_cities.append(origin)
    coordinates = [config.CITY_COORDINATES[city] for city in route_cities]
    fig = go.Figure(
        go.Scattergeo(
            lat=[point[0] for point in coordinates],
            lon=[point[1] for point in coordinates],
            mode="lines+markers+text",
            text=route_cities,
            textposition="top center",
            line=dict(width=3, color="#4C78A8"),
            marker=dict(size=10, color="#C0392B"),
        )
    )
    fig.update_geos(
        scope="asia",
        fitbounds="locations",
        showcountries=True,
        countrycolor="#CCCCCC",
        showland=True,
        landcolor="#F7F7F7",
    )
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def page_explorer(frame: pd.DataFrame) -> None:
    st.title("Data Explorer")
    st.caption("Filter the 5,000-trip fleet log and download the result.")
    c1, c2, c3 = st.columns(3)
    status = c1.multiselect(
        "Delivery status",
        ["On_Time", "Delayed", "Failed"],
        default=["On_Time", "Delayed", "Failed"],
    )
    vehicle = c2.selectbox("Vehicle", ["All", *sorted(frame.vehicle_id.unique())])
    driver = c3.selectbox("Driver", ["All", *sorted(frame.driver_id.unique())])

    c4, c5, c6 = st.columns(3)
    origin = c4.selectbox("Origin", ["All", *config.CITIES])
    destination = c5.selectbox("Destination", ["All", *config.CITIES])
    date_range = c6.date_input(
        "Date range",
        value=(frame.trip_start_time.min().date(), frame.trip_start_time.max().date()),
    )

    filtered = frame.copy()
    if status:
        filtered = filtered[filtered.delivery_status.isin(status)]
    if vehicle != "All":
        filtered = filtered[filtered.vehicle_id == vehicle]
    if driver != "All":
        filtered = filtered[filtered.driver_id == driver]
    if origin != "All":
        filtered = filtered[filtered.origin == origin]
    if destination != "All":
        filtered = filtered[filtered.destination == destination]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        filtered = filtered[
            (filtered.trip_start_time >= start) & (filtered.trip_start_time < end)
        ]

    st.metric("Matching trips", f"{len(filtered):,}")
    columns = [
        "trip_id",
        "trip_start_time",
        "vehicle_id",
        "driver_id",
        "origin",
        "destination",
        "distance_km",
        "fuel_L",
        "kmpl",
        "delivery_status",
        "driver_score",
        "vehicle_class",
        "is_anomaly",
    ]
    st.dataframe(filtered[columns], use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered CSV",
        filtered[columns].to_csv(index=False).encode("utf-8"),
        file_name="fleet_trips_filtered.csv",
        mime="text/csv",
    )


def main() -> None:
    bootstrap()
    frame = load_trips()
    scores = load_scores()

    st.sidebar.title("🚚 Fleet Command Centre")
    st.sidebar.caption("HCLTech Career Shaper · Project 24")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Delivery Performance",
            "Fuel & Anomalies",
            "Driver Behaviour",
            "Route Optimiser",
            "Data Explorer",
        ],
    )
    st.sidebar.divider()
    st.sidebar.metric("Trips loaded", f"{len(frame):,}")
    st.sidebar.metric("On-time", f"{(frame.delivery_status == 'On_Time').mean():.1%}")
    st.sidebar.caption(
        "Synthetic/anonymised academic dataset. No real personal data. "
        "Driver scores are decision-support only."
    )

    if page == "Overview":
        page_overview(frame)
    elif page == "Delivery Performance":
        page_delivery(frame)
    elif page == "Fuel & Anomalies":
        page_fuel(frame)
    elif page == "Driver Behaviour":
        page_drivers(frame, scores)
    elif page == "Route Optimiser":
        page_routes()
    else:
        page_explorer(frame)


if __name__ == "__main__":
    main()
