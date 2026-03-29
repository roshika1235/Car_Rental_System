import pandas as pd
import pytest
import numpy as np

from cleaning.cleaner import (
    clean_vehicle_id,
    clean_odo_start,
    clean_odo_end,
    prepare_odo,
    clean_fuel_level,
    clean_rate,
    remove_duplicate_reservation,
    clean_city,
    validate_trip_time,
    clean_payment,
    mileage_sanity_check,
    fuel_sanity_check,
    trip_overlap_check,
    damage_analysis,
    clean_driver_license,
    clean_promo_code,
    smooth_gps,
    normalize_harsh_events,
    redact_pii,
    prepare_rate_plan_data,
    validate_gst,
    normalize_time,
    clean_timestamps
)


# ---------- SAMPLE DATA ----------
def sample_df():
    return pd.DataFrame({
        "Vehicle_ID": [" ab12 ", "cd34"],

        "Odo_Start_km": ["1,000km", "2000km"],
        "Odo_End_km": ["1,500km", "2500km"],

        "Fuel_Level": ["50%", "0.5"],

        "Rate": ["₹1,000/day", "₹2000/day"],

        "Reservation_ID": ["R1", "R1"],  # duplicate

        "City": ["delhi", "blr"],

        "Payment": ["card", "upi"],
        "Booking_TS": ["2024-01-01 10:00:00", "2024-01-02 10:00:00"],
        "Pickup_TS": ["2024-01-01 10:00:00", "2024-01-02 10:00:00"],
        "Return_TS": ["2024-01-01 12:00:00", "2024-01-02 09:00:00"],

        "Fuel_Fraction": [0.5, 0.9],  # for fuel test
        "Damage_Flag": [1, 0],
        "Driver_License": ["DL12345", None],
        "Promo_Code": ["SAVE10", None],

        "GPS_Lat": [17.1234567,17.1234],
        "GPS_Lon": [17.1234,78.9877],

        "Harsh_Events": [3, None],

        "Notes": ["Call me at 9876543210", "No issues"],
        "Customer_Feedback": ["email test@gmail.com", None],

        "Rate_Plan": ["std", "premium"],
        "Daily_Rate":[None,2600],

        "GST_Amount": [180, None],          # one missing
        "Total_Amount": [1180, 2100] 

    })


# ---------- TESTS ----------
# ---- 1. Vehicle_ID trimming and canonical case.----------
def test_vehicle_id():
    df = sample_df()
    df = clean_vehicle_id(df)
    assert df['Vehicle_ID'][0] == "AB12"

# ---- 2. timestamp validation----------
def test_normalize_time():

    ts = pd.to_datetime("2024-01-01 10:70:80", errors='coerce')

    result = normalize_time(ts)

    # check not null
    assert result is not None


# ---------- test clean_timestamps ----------
def test_clean_timestamps():

    df = sample_df()

    df = clean_timestamps(df)

    # check columns exist
    assert "Pickup_TS" in df.columns
    assert "Booking_TS" in df.columns
    assert "Return_TS" in df.columns

    # check no null pickup
    assert df["Pickup_TS"].isnull().sum() == 0

    # check datetime type
    assert pd.api.types.is_datetime64_any_dtype(df["Pickup_TS"])

# ---- 3. Odometer numeric extraction and unit unification (km).----------
def test_odo_clean():
    df = sample_df()
    df = clean_odo_start(df)
    df = clean_odo_end(df)

    assert df['Odo_Start_km'][0] == 1000
    assert df['Odo_End_km'][0] == 1500


def test_prepare_odo():
    df = sample_df()
    df = prepare_odo(df)

    assert "Odo_Start_km" in df.columns
    assert "Odo_End_km" in df.columns


# ---- 4. Fuel level normalization (50%→0.5).----------
def test_fuel():
    df = sample_df()

    # remove if exists (safety)
    if 'Fuel_Fraction' in df.columns:
        df = df.drop(columns=['Fuel_Fraction'])

    df = clean_fuel_level(df)

    assert 0 <= df['Fuel_Fraction'].iloc[0] <= 1


# ---- 5. Rate parsing to numeric daily rate; currency normalization.-----
def test_rate():
    df = sample_df()

    # remove existing Daily_Rate (important fix)
    if 'Daily_Rate' in df.columns:
        df = df.drop(columns=['Daily_Rate'])

    df = clean_rate(df)

    assert df['Daily_Rate'].iloc[0] == 1000

# ---- 6. City normalization to canonical names.-----

def test_duplicates():
    df = sample_df()
    df = remove_duplicate_reservation(df)

    assert df['Reservation_ID'].duplicated().sum() == 0

# ----7. Duplicate reservation dedup (same ID).-----

def test_city():
    df = sample_df()
    df = clean_city(df)

    assert df['City'][0] == "Delhi"

# ----8. Return before pickup rule validation.-----

def test_payment():
    df = sample_df()
    df = clean_payment(df)

    assert df['Payment'][0] == "CARD"

# ----9. Payment method standardization (UPI/CARD/CASH/WALLET).-----

def test_trip_time():
    df = sample_df()
    df = validate_trip_time(df)

    assert all(df['Return_TS'] >= df['Pickup_TS'])

# ---- 10. Mileage sanity check (End ≥ Start).-----

def test_mileage_flag():
    df = sample_df()
    df = clean_odo_start(df)
    df = clean_odo_end(df)

    df = mileage_sanity_check(df)

    assert "mileage_flag" in df.columns
    assert df['mileage_flag'][0] == "VALID"

# ---- 11. Refueling event detection vs fuel change and distance.-----

def test_fuel_flag():
    df = sample_df()
    df = clean_odo_start(df)
    df = clean_odo_end(df)

    df['Fuel_Fraction'] = [0.1, 0.9]

    df = fuel_sanity_check(df)

    assert "fuel_flag" in df.columns

# ---- 12. Vehicle availability overlap checks.-----

def test_overlap():
    df = sample_df()

    df = trip_overlap_check(df)

    assert "overlap_flag" in df.columns


# ---- 13. Damage/incident log linkage to reservation.-----

def test_damage_analysis():
    df = sample_df()

    damage_cases, damage_rate = damage_analysis(df)

    # check filtered rows
    assert len(damage_cases) == 1
    assert damage_cases['Damage_Flag'].iloc[0] == 1

    # check rate calculation
    assert round(damage_rate[1], 2) == 50.0
    assert round(damage_rate[0], 2) == 50.0


# ---- 14. Driver license masking and validation (if present).-----

def test_driver_license():
    df = sample_df()

    df = clean_driver_license(df)

    # check masking
    assert df["Driver_License"].iloc[0] == "DL***45"

    # check missing handled
    assert df["Driver_License"].iloc[1] == "UNKNOWN"

    # check validation column exists
    assert "License_Valid" in df.columns

    # check validation logic
    assert df["License_Valid"].iloc[0] == True
    assert df["License_Valid"].iloc[1] == False


# ---- 15. Promo/coupon code validation & expiry checks.-----

def test_promo_code():
    df = sample_df()

    df = clean_promo_code(df)

    # ✅ column created
    assert "Promo_Valid" in df.columns

    # ✅ valid promo
    assert df["Promo_Valid"].iloc[0] == True

    # ✅ missing handled
    assert df["Promo_Code"].iloc[1] == "NO_PROMO"
    assert df["Promo_Valid"].iloc[1] == False


# ----16. Telematics GPS join and jitter smoothing.-----

def test_smooth_gps_rounding():

    df = sample_df()
    df = smooth_gps(df)

    # check row 0
    assert df["GPS_Lat_Smoothed"].iloc[0] == 17.1235
    assert df["GPS_Lon_Smoothed"].iloc[0] == 17.1234

    # check row 1
    assert df["GPS_Lat_Smoothed"].iloc[1] == 17.1234
    assert df["GPS_Lon_Smoothed"].iloc[1] == 78.9877

# ---- 17. Speeding/harsh events normalization from telematics.-----

def test_harsh_events():

    df = sample_df()
    df = normalize_harsh_events(df)

    # row 0: 3 → 1
    assert df["Harsh_Events"].iloc[0] == 1

    # row 1: None → 0
    assert df["Harsh_Events"].iloc[1] == 0


# ---- 18. PII redaction in notes and feedback.-----

def test_redact_pii():

    df = sample_df()
    df = redact_pii(df)

    # ✅ phone masked
    assert "**********" in df["Notes"].iloc[0]

    # ✅ email masked
    assert "*****@*****" in df["Customer_Feedback"].iloc[0]

    # ✅ no PII remains unchanged
    assert df["Notes"].iloc[1] == "No issues"

    # ✅ None stays None
    assert pd.isnull(df["Customer_Feedback"].iloc[1])


# ---- 19. Rate plan mapping to master tariffs.-----


def test_prepare_rate_plan_data():
    df = sample_df()
    df = prepare_rate_plan_data(df)

    assert df['Rate_Plan'].iloc[0] == "standard"
    assert df['Daily_Rate'].iloc[0] == 1500
    assert df['Daily_Rate'].iloc[1] == 2600

# ---- 20. Tax/GST computation validation.-----

def test_validate_gst():
    df = sample_df()

    # 🔥 override values for this test only
    df['Daily_Rate'] = [1000, 2000]
    df['GST_Amount'] = [180, None]
    df['Total_Amount'] = [1180, 2100]

    df = validate_gst(df)

    assert "GST_Error" in df.columns
    assert df["GST_Error"].iloc[0] == False
    assert df["GST_Error"].iloc[1] == True





    


