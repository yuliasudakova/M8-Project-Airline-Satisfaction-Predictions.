import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Airline Passenger Satisfaction",
    page_icon="✈️",
    layout="wide"
)

# -----------------------------
# Helpers
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace("-", "_").replace(" ", "_")
        for col in df.columns
    ]
    return df


def map_satisfaction_target(series: pd.Series) -> pd.Series:
    if series.dtype != "object":
        return series

    cleaned = series.astype(str).str.strip().str.lower()

    mapping = {
        "neutral or dissatisfied": 0,
        "neutral_or_dissatisfied": 0,
        "dissatisfied": 0,
        "satisfied": 1
    }

    mapped = cleaned.map(mapping)

    if mapped.isna().any():
        unknown_values = cleaned[mapped.isna()].unique().tolist()
        raise ValueError(f"Unexpected target values in 'satisfaction': {unknown_values}")

    return mapped


# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Data/cleaned_airline_passenger_satisfaction.csv")
        df = normalize_columns(df)
        df = df.loc[:, ~df.columns.str.contains("^unnamed", case=False, regex=True)]
        return df
    except FileNotFoundError:
        st.error("Dataset file not found: Data/cleaned_airline_passenger_satisfaction.csv")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


air = load_data()

# -----------------------------
# Train model inside app
# -----------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    if df.empty:
        return None, None

    data = df.copy()

    if "id" in data.columns:
        data = data.drop(columns=["id"])

    if "satisfaction" not in data.columns:
        raise ValueError("Target column 'satisfaction' not found in dataset.")

    y = map_satisfaction_target(data["satisfaction"])
    X = data.drop(columns=["satisfaction"])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=500))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc


try:
    model, logreg_accuracy = train_model(air)
except Exception as e:
    model, logreg_accuracy = None, None
    st.error(f"Model training failed: {e}")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Airline Passenger Satisfaction")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "EDA Insights",
        "Model Performance",
        "Make a Prediction",
        "Recommendations"
    ]
)

# =============================
# HOME
# =============================
if page == "Home":
    st.title("Airline Passenger Satisfaction Analysis")

    st.write(
        """
        This interactive dashboard explores factors that influence airline passenger satisfaction
        and allows users to predict satisfaction using a trained machine learning model.

        The project includes:
        - Exploratory Data Analysis (EDA)
        - Logistic Regression modeling
        - Interactive passenger satisfaction predictions
        """
    )

    if not air.empty:
        st.subheader("Dataset Preview")
        st.dataframe(air.head())

        st.subheader("Dataset Shape")
        st.write(f"Rows: {air.shape[0]}")
        st.write(f"Columns: {air.shape[1]}")
    else:
        st.warning("Dataset could not be loaded.")

# =============================
# EDA INSIGHTS
# =============================
elif page == "EDA Insights":
    st.title("EDA Insights")

    if air.empty:
        st.warning("Dataset is not available.")
    else:
        if "satisfaction" in air.columns:
            st.subheader("Satisfaction Distribution")

            counts = air["satisfaction"].value_counts()

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(counts.index.astype(str), counts.values)
            ax.set_title("Passenger Satisfaction")
            ax.set_xlabel("Satisfaction")
            ax.set_ylabel("Count")
            plt.xticks(rotation=15)
            st.pyplot(fig)

        st.subheader("Service Ratings vs Satisfaction")

        service_cols = [
            c for c in [
                "seat_comfort",
                "inflight_entertainment",
                "on_board_service",
                "cleanliness",
                "inflight_wifi_service",
                "food_and_drink",
                "gate_location"
            ]
            if c in air.columns
        ]

        if service_cols and "satisfaction" in air.columns:
            chosen = st.selectbox("Pick a service feature", service_cols)

            fig, ax = plt.subplots(figsize=(7, 4))

            groups = []
            labels = []
            for label, subset in air.groupby("satisfaction"):
                groups.append(subset[chosen].dropna())
                labels.append(str(label))

            if groups:
                ax.boxplot(groups, tick_labels=labels)
                ax.set_title(f"{chosen} vs Satisfaction")
                ax.set_xlabel("Satisfaction")
                ax.set_ylabel(chosen)
                st.pyplot(fig)
        else:
            st.info("Required service columns are not available.")

# =============================
# MODEL PERFORMANCE
# =============================
elif page == "Model Performance":
    st.title("Model Evaluation")

    if logreg_accuracy is None:
        st.warning("Model accuracy is not available because model training failed.")
    else:
        results = pd.DataFrame({
            "Model": ["Logistic Regression", "KNN", "Random Forest"],
            "Accuracy": [round(logreg_accuracy, 3), 0.928, 0.964]
        })

        st.table(results)
        st.success("Random Forest achieved the best performance in the notebook results.")

        st.info(
            "In this app, Logistic Regression is trained dynamically from the dataset "
            "and used for the live prediction page."
        )

# =============================
# MAKE A PREDICTION
# =============================
elif page == "Make a Prediction":
    st.title("Make a Prediction")
    st.write("Enter passenger details and service ratings to predict satisfaction.")

    if model is None:
        st.warning("Prediction model is not available.")
    else:
        left, right = st.columns(2)

        with left:
            gender = st.selectbox("Gender", ["Female", "Male"])
            customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
            age = st.slider("Age", 7, 85, 30)
            type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
            flight_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
            flight_distance = st.slider("Flight Distance", 50, 5000, 800)
            departure_delay = st.number_input(
                "Departure delay (minutes)", min_value=0.0, max_value=5000.0, value=0.0
            )
            arrival_delay = st.number_input(
                "Arrival delay (minutes)", min_value=0.0, max_value=5000.0, value=0.0
            )

        with right:
            st.write("Service ratings (0–5)")
            inflight_wifi_service = st.slider("Inflight WiFi", 0, 5, 3)
            departure_arrival_time_convenient = st.slider("Time Convenient", 0, 5, 3)
            ease_of_online_booking = st.slider("Ease of Online Booking", 0, 5, 3)
            online_boarding = st.slider("Online Boarding", 0, 5, 3)
            seat_comfort = st.slider("Seat Comfort", 0, 5, 3)
            inflight_entertainment = st.slider("Inflight Entertainment", 0, 5, 3)
            on_board_service = st.slider("On-board Service", 0, 5, 3)
            leg_room_service = st.slider("Leg Room", 0, 5, 3)
            baggage_handling = st.slider("Baggage Handling", 0, 5, 3)
            checkin_service = st.slider("Check-in Service", 0, 5, 3)
            inflight_service = st.slider("Inflight Service", 0, 5, 3)
            cleanliness = st.slider("Cleanliness", 0, 5, 3)
            gate_location = st.slider("Gate Location", 0, 5, 3)
            food_and_drink = st.slider("Food and Drink", 0, 5, 3)

        if st.button("Predict Satisfaction"):
            try:
                row = pd.DataFrame([{
                    "gender": gender,
                    "customer_type": customer_type,
                    "age": age,
                    "type_of_travel": type_of_travel,
                    "class": flight_class,
                    "flight_distance": flight_distance,
                    "inflight_wifi_service": inflight_wifi_service,
                    "departure_arrival_time_convenient": departure_arrival_time_convenient,
                    "ease_of_online_booking": ease_of_online_booking,
                    "online_boarding": online_boarding,
                    "seat_comfort": seat_comfort,
                    "inflight_entertainment": inflight_entertainment,
                    "on_board_service": on_board_service,
                    "leg_room_service": leg_room_service,
                    "baggage_handling": baggage_handling,
                    "checkin_service": checkin_service,
                    "inflight_service": inflight_service,
                    "cleanliness": cleanliness,
                    "departure_delay_in_minutes": departure_delay,
                    "arrival_delay_in_minutes": arrival_delay,
                    "gate_location": gate_location,
                    "food_and_drink": food_and_drink
                }])

                row = normalize_columns(row)

                pred = model.predict(row)[0]
                proba = model.predict_proba(row)[0][1]

                if pred == 1:
                    st.success("✅ Prediction: Satisfied")
                else:
                    st.error("⚠️ Prediction: Neutral / Dissatisfied")

                st.write(f"Probability of being satisfied: {proba:.2%}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# =============================
# RECOMMENDATIONS
# =============================
elif page == "Recommendations":
    st.title("Business Recommendations")

    st.write(
        """
        Based on the analysis and machine learning results, airlines can improve
        passenger satisfaction by focusing on the following factors:
        """
    )

    st.markdown(
        """
        **1. Improve seat comfort**  
        Comfortable seating strongly increases satisfaction.

        **2. Enhance inflight entertainment**  
        Entertainment options significantly improve passenger experience.

        **3. Maintain high cleanliness standards**  
        Clean cabins positively affect passenger perception.

        **4. Reduce delays**  
        Flight delays negatively impact satisfaction.

        **5. Improve service quality**  
        Professional and friendly service improves overall experience.
        """
    )
    