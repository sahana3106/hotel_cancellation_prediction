import streamlit as st
import pandas as pd
import joblib
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

# Function used when the model pipeline was trained
def to_string(x):
    return x.astype(str)
# ============================================================
# LOAD MODEL + DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("hotel_cancellation_model_deploy.pkl")

model = load_model()
# Put your original hotel booking dataset in the same folder
# as app.py and rename it to hotel_booking.csv
df_dashboard = pd.read_csv("hotel_bookings.csv")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hotel Cancellation Analytics",
    page_icon="🏨",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🏨 Hotel Cancellation Analytics")

    page = st.radio(
        "🧭 Navigate",
        ["📊 Dashboard", "🔮 Prediction"]
    )

    st.divider()

    st.header("ℹ️ About this Project")

    st.write(
        "This application analyzes hotel booking patterns "
        "and predicts booking cancellations using a "
        "Random Forest classification model."
    )

    st.write("🌲 **Model:** Random Forest")
    st.write("🎯 **Task:** Binary Classification")
    st.write("🚫 **Target:** Booking Cancellation")

    st.divider()

    st.caption(
        "Developed as an end-to-end machine learning project."
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "📊 Dashboard":

    st.title("📊 Hotel Booking Analytics Dashboard")

    st.write(
        "Explore booking patterns and factors associated "
        "with hotel cancellations."
    )

    # --------------------------------------------------------
    # KPI METRICS
    # --------------------------------------------------------

    total_bookings = len(df_dashboard)

    cancelled_bookings = df_dashboard["is_canceled"].sum()

    cancellation_rate = (
        cancelled_bookings / total_bookings
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📚 Total Bookings",
            f"{total_bookings:,}"
        )

    with col2:
        st.metric(
            "❌ Cancelled Bookings",
            f"{cancelled_bookings:,}"
        )

    with col3:
        st.metric(
            "📉 Cancellation Rate",
            f"{cancellation_rate:.1%}"
        )

    st.divider()

    # --------------------------------------------------------
    # HOTEL CANCELLATION
    # --------------------------------------------------------

    st.subheader("🏨 Cancellation Rate by Hotel")

    hotel_cancel = (
        df_dashboard
        .groupby("hotel")["is_canceled"]
        .mean()
        .mul(100)
    )

    st.bar_chart(hotel_cancel)

    # --------------------------------------------------------
    # DEPOSIT TYPE
    # --------------------------------------------------------

    st.subheader("💳 Cancellation Rate by Deposit Type")

    deposit_cancel = (
        df_dashboard
        .groupby("deposit_type")["is_canceled"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    st.bar_chart(deposit_cancel)

    # --------------------------------------------------------
    # MARKET SEGMENT
    # --------------------------------------------------------

    st.subheader("📋 Cancellation Rate by Market Segment")

    segment_cancel = (
        df_dashboard
        .groupby("market_segment")["is_canceled"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    st.bar_chart(segment_cancel)

    # --------------------------------------------------------
    # MONTHLY CANCELLATION
    # --------------------------------------------------------

    st.subheader("📅 Cancellation Rate by Arrival Month")

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    monthly_cancel = (
        df_dashboard
        .groupby("arrival_date_month")["is_canceled"]
        .mean()
        .mul(100)
        .reindex(month_order)
    )

    st.line_chart(monthly_cancel)

    # --------------------------------------------------------
    # LEAD TIME
    # --------------------------------------------------------

    st.subheader("⏳ Cancellation Rate by Lead Time")

    lead_bins = [
        0,
        30,
        60,
        90,
        180,
        365,
        float("inf")
    ]

    lead_labels = [
        "0–30",
        "31–60",
        "61–90",
        "91–180",
        "181–365",
        "365+"
    ]

    dashboard_lead = df_dashboard.copy()

    dashboard_lead["lead_time_group"] = pd.cut(
        dashboard_lead["lead_time"],
        bins=lead_bins,
        labels=lead_labels
    )

    lead_cancel = (
        dashboard_lead
        .groupby(
            "lead_time_group",
            observed=False
        )["is_canceled"]
        .mean()
        .mul(100)
    )

    st.bar_chart(lead_cancel)

    # --------------------------------------------------------
    # ADR
    # --------------------------------------------------------

    st.subheader("💰 Average ADR by Cancellation Status")

    adr_cancel = (
        df_dashboard
        .groupby("is_canceled")["adr"]
        .mean()
        .rename({
            0: "Not Cancelled",
            1: "Cancelled"
        })
    )

    st.bar_chart(adr_cancel)

    # --------------------------------------------------------
    # SPECIAL REQUESTS
    # --------------------------------------------------------

    st.subheader(
        "⭐ Cancellation Rate by Special Requests"
    )

    special_requests_cancel = (
        df_dashboard
        .groupby("total_of_special_requests")["is_canceled"]
        .mean()
        .mul(100)
    )

    st.line_chart(special_requests_cancel)

    st.divider()

    st.info(
        "The dashboard summarizes patterns found during "
        "exploratory data analysis. These relationships "
        "describe the historical dataset and should not "
        "be interpreted as causal effects."
    )
    # --------------------------------------------------------
    # MODEL FEATURE IMPORTANCE
    # --------------------------------------------------------

    # --------------------------------------------------------
# MODEL FEATURE IMPORTANCE
# --------------------------------------------------------

st.subheader("🌲 Top Features Influencing the Model")

# Get fitted preprocessing and Random Forest model
preprocessor = model.named_steps["preprocessor"]
rf_model = model.named_steps["model"]

# Get feature names manually from each transformer
feature_names = []

for name, transformer, columns in preprocessor.transformers_:

    if name == "num":

        # Numerical features keep their original names
        feature_names.extend(columns)

    elif name == "cat":

        # Get the categorical pipeline
        cat_pipeline = transformer

        # Get the OneHotEncoder
        onehot = cat_pipeline.named_steps["onehot"]

        # Get names produced by OneHotEncoder
        cat_feature_names = onehot.get_feature_names_out(columns)

        feature_names.extend(cat_feature_names)


# Get Random Forest feature importance
importances = rf_model.feature_importances_

# Create DataFrame
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort by importance
feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(15)
    .sort_values(
        "Importance",
        ascending=True
    )
)

# Display
st.bar_chart(
    feature_importance.set_index("Feature")
)
# ============================================================
# PREDICTION PAGE
# ============================================================

if page == "🔮 Prediction":

    st.title("🔮 Hotel Booking Cancellation Predictor")

    st.write(
        "Enter the booking details below to estimate the "
        "probability of cancellation using a trained "
        "Random Forest machine learning model."
    )

    st.info(
        "💡 The prediction is based on patterns learned "
        "from historical hotel booking data."
    )

    # --------------------------------------------------------
    # BOOKING INFORMATION
    # --------------------------------------------------------

    st.header("📅 Booking Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        hotel = st.selectbox(
            "🏨 Hotel",
            [
                "Resort Hotel",
                "City Hotel"
            ]
        )

    with col2:

        arrival_date = st.date_input(
            "📅 Arrival Date",
            value=date(2017, 7, 1)
        )

    with col3:

        lead_time = st.number_input(
            "⏳ Lead Time (days)",
            min_value=0,
            max_value=1000,
            value=100
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        stays_in_weekend_nights = st.number_input(
            "🌙 Weekend Nights",
            min_value=0,
            value=1
        )

    with col2:

        stays_in_week_nights = st.number_input(
            "🛏️ Week Nights",
            min_value=0,
            value=2
        )

    with col3:

        adults = st.number_input(
            "👨‍👩‍👧 Adults",
            min_value=0,
            value=2
        )

    with col4:

        children = st.number_input(
            "🧒 Children",
            min_value=0.0,
            value=0.0
        )

    babies = st.number_input(
        "👶 Babies",
        min_value=0,
        value=0
    )

    meal = st.selectbox(
        "🍽️ Meal",
        [
            "BB",
            "HB",
            "SC",
            "Undefined",
            "FB"
        ]
    )

    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    st.header("👤 Customer Information")

    country_options = [
        'ABW', 'AGO', 'AIA', 'ALB', 'AND', 'ARE',
        'ARG', 'ARM', 'ASM', 'ATA', 'ATF', 'AUS',
        'AUT', 'AZE', 'BEL', 'BEN', 'BFA', 'BGD',
        'BGR', 'BHR', 'BHS', 'BIH', 'BLR', 'BOL',
        'BRA', 'BRB', 'CAF', 'CHE', 'CHL', 'CHN',
        'CIV', 'CMR', 'CN', 'COL', 'COM', 'CPV',
        'CRI', 'CUB', 'CYM', 'CYP', 'CZE', 'DEU',
        'DJI', 'DMA', 'DNK', 'DOM', 'DZA', 'ECU',
        'EGY', 'ESP', 'EST', 'ETH', 'FIN', 'FJI',
        'FRA', 'FRO', 'GAB', 'GBR', 'GEO', 'GGY',
        'GHA', 'GIB', 'GLP', 'GNB', 'GRC', 'GTM',
        'HKG', 'HRV', 'HUN', 'IDN', 'IMN', 'IND',
        'IRL', 'IRN', 'IRQ', 'ISL', 'ISR', 'ITA',
        'JAM', 'JEY', 'JOR', 'JPN', 'KAZ', 'KEN',
        'KHM', 'KNA', 'KOR', 'KWT', 'LAO', 'LBN',
        'LBY', 'LIE', 'LKA', 'LTU', 'LUX', 'LVA',
        'MAC', 'MAR', 'MCO', 'MDG', 'MDV', 'MEX',
        'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MOZ',
        'MRT', 'MUS', 'MWI', 'MYS', 'MYT', 'NAM',
        'NCL', 'NGA', 'NLD', 'NOR', 'NPL', 'NZL',
        'OMN', 'PAK', 'PAN', 'PER', 'PHL', 'PLW',
        'POL', 'PRI', 'PRT', 'PRY', 'PYF', 'QAT',
        'ROU', 'RUS', 'RWA', 'SAU', 'SDN', 'SEN',
        'SGP', 'SLE', 'SLV', 'SMR', 'SRB', 'STP',
        'SUR', 'SVK', 'SVN', 'SWE', 'SYC', 'SYR',
        'TGO', 'THA', 'TJK', 'TMP', 'TUN', 'TUR',
        'TWN', 'TZA', 'UGA', 'UKR', 'UMI', 'URY',
        'USA', 'UZB', 'VEN', 'VGB', 'VNM', 'ZAF',
        'ZMB', 'ZWE'
    ]

    col1, col2 = st.columns(2)

    with col1:

        country = st.selectbox(
            "🌍 Country",
            country_options
        )

    with col2:

        is_repeated_guest = st.selectbox(
            "🔁 Repeated Guest?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

    col1, col2 = st.columns(2)

    with col1:

        previous_cancellations = st.number_input(
            "❌ Previous Cancellations",
            min_value=0,
            value=0
        )

    with col2:

        previous_bookings_not_canceled = st.number_input(
            "✅ Previous Bookings Not Cancelled",
            min_value=0,
            value=0
        )

    # --------------------------------------------------------
    # BOOKING INFORMATION
    # --------------------------------------------------------

    st.header("📋 Booking Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        market_segment = st.selectbox(
            "📊 Market Segment",
            [
                "Groups",
                "Online TA",
                "Corporate",
                "Offline TA/TO",
                "Direct",
                "Complementary",
                "Aviation"
            ]
        )

    with col2:

        distribution_channel = st.selectbox(
            "📡 Distribution Channel",
            [
                "TA/TO",
                "Corporate",
                "Direct",
                "GDS"
            ]
        )

    with col3:

        deposit_type = st.selectbox(
            "💳 Deposit Type",
            [
                "No Deposit",
                "Non Refund",
                "Refundable"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        reserved_room_type = st.selectbox(
            "🛏️ Reserved Room Type",
            [
                "A", "B", "C", "D", "E",
                "F", "G", "H", "L", "P"
            ]
        )

    with col2:

        assigned_room_type = st.selectbox(
            "🔑 Assigned Room Type",
            [
                "A", "B", "C", "D", "E",
                "F", "G", "H", "L", "P"
            ]
        )

    booking_changes = st.number_input(
        "🔄 Booking Changes",
        min_value=0,
        value=0
    )

    # --------------------------------------------------------
    # ADDITIONAL DETAILS
    # --------------------------------------------------------

    st.header("💰 Additional Booking Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        days_in_waiting_list = st.number_input(
            "⏱️ Days in Waiting List",
            min_value=0,
            value=0
        )

    with col2:

        adr = st.number_input(
            "💰 Average Daily Rate (ADR)",
            min_value=0.0,
            value=100.0
        )

    with col3:

        required_car_parking_spaces = st.number_input(
            "🚗 Required Car Parking Spaces",
            min_value=0,
            max_value=8,
            value=0
        )

    col1, col2 = st.columns(2)

    with col1:

        total_of_special_requests = st.number_input(
            "⭐ Total Special Requests",
            min_value=0,
            max_value=5,
            value=0
        )

    with col2:

        customer_type = st.selectbox(
            "👤 Customer Type",
            [
                "Transient",
                "Transient-Party",
                "Contract",
                "Group"
            ]
        )

    # --------------------------------------------------------
    # AGENT + COMPANY
    # --------------------------------------------------------

    st.header("🏢 Agent & Company Information")

    col1, col2 = st.columns(2)

    with col1:

        agent_input = st.number_input(
            "🧑‍💼 Agent ID (leave 0 if unknown)",
            min_value=0.0,
            value=0.0
        )

    with col2:

        has_company = st.selectbox(
            "🏢 Company Associated?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

    # --------------------------------------------------------
    # DERIVED FEATURES
    # --------------------------------------------------------

    arrival_date_year = arrival_date.year

    arrival_date_month = arrival_date.strftime("%B")

    arrival_date_day_of_month = arrival_date.day

    arrival_date_week_number = arrival_date.isocalendar().week

    total_night_stay = (
        stays_in_weekend_nights
        + stays_in_week_nights
    )

    total_guest = (
        adults
        + children
        + babies
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Cancellation",
        use_container_width=True
    ):

        agent = (
            None
            if agent_input == 0
            else agent_input
        )

        input_data = pd.DataFrame([{

            "hotel": hotel,

            "lead_time": lead_time,

            "arrival_date_year":
                arrival_date_year,

            "arrival_date_month":
                arrival_date_month,

            "arrival_date_week_number":
                arrival_date_week_number,

            "arrival_date_day_of_month":
                arrival_date_day_of_month,

            "stays_in_weekend_nights":
                stays_in_weekend_nights,

            "stays_in_week_nights":
                stays_in_week_nights,

            "adults": adults,

            "children": children,

            "babies": babies,

            "meal": meal,

            "country": country,

            "market_segment":
                market_segment,

            "distribution_channel":
                distribution_channel,

            "is_repeated_guest":
                is_repeated_guest,

            "previous_cancellations":
                previous_cancellations,

            "previous_bookings_not_canceled":
                previous_bookings_not_canceled,

            "reserved_room_type":
                reserved_room_type,

            "assigned_room_type":
                assigned_room_type,

            "booking_changes":
                booking_changes,

            "deposit_type":
                deposit_type,

            "agent": agent,

            "days_in_waiting_list":
                days_in_waiting_list,

            "customer_type":
                customer_type,

            "adr": adr,

            "required_car_parking_spaces":
                required_car_parking_spaces,

            "total_of_special_requests":
                total_of_special_requests,

            "total_night_stay":
                total_night_stay,

            "total_guest":
                total_guest,

            "has_company":
                has_company
        }])

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader("🔮 Prediction Result")

        not_cancel_probability = 1 - probability

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "📈 Cancellation Probability",
                f"{probability:.1%}"
            )

        with col2:

            st.metric(
                "📉 Not-Cancellation Probability",
                f"{not_cancel_probability:.1%}"
            )

        st.divider()

        if prediction == 1:

            if probability >= 0.75:
                risk = "🔴 High Risk"

            elif probability >= 0.50:
                risk = "🟠 Medium Risk"

            else:
                risk = "🟡 Low Risk"

            st.error(
                "⚠️ This booking is likely to be cancelled."
            )

        else:

            if probability < 0.25:
                risk = "🟢 Very Low Risk"

            elif probability < 0.50:
                risk = "🟡 Low Risk"

            else:
                risk = "🟠 Medium Risk"

            st.success(
                "✅ This booking is unlikely to be cancelled."
            )

        st.write(
            f"### Risk Level: {risk}"
        )