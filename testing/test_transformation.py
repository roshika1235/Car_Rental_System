# import pandas
import pandas as pd

# import numpy
import numpy as np

# import all transformation functions
from Transformations.transformations import *


# create a sample dataframe for testing
def sample_df():

    # returning sample dataset
    return pd.DataFrame({
        "Reservation_ID": ["RES1", "RES2", "RES3"],
        "Customer_ID_x": ["C1", "C2", "C3"],
        "Vehicle_ID": ["V1", "V1", "V2"],
        "Vehicle_Class": ["X", "Y", "X"],

        "Booking_TS": ["2023-12-31 10:00", "2024-01-01 10:00", "2024-01-02 10:00"],
        "Pickup_TS": ["2024-01-01 10:00", "2024-01-02 10:00", "2024-01-03 10:00"],
        "Return_TS": ["2024-01-01 18:00", "2024-01-02 18:00", "2024-01-03 18:00"],

        "City": ["A", "A", "B"],

        "Pickup_Lat": [10.0, 20.0, 30.0],
        "Pickup_Lon": [30.0, 40.0, 50.0],
        "Drop_Lat": [15.0, 25.0, 35.0],
        "Drop_Lon": [35.0, 45.0, 55.0],

        "Odo_Start_km": [100, 200, 300],
        "Odo_End_km": [200, 300, 450],

        "Fuel_Fraction": [0.5, 0.1, 0.3],

        "Damage_Flag": [0, 1, 0],

        "GPS_Lat": [10.1, 20.1, 30.1],
        "GPS_Lon": [30.1, 40.1, 50.1],

        "Max_Speed_kmh": [60, 80, 70],
        "Harsh_Events": [1, 2, 0],

        "Payment": ["UPI", "CARD", "CASH"],
        "Promo_Code": ["P1", "P2", "P3"],

        "GST_Amount": [50.0, 100.0, 80.0],
        "Total_Amount": [1050.0, 2100.0, 3080.0],

        "Rate_Plan": ["R1", "R2", "R1"],
        "Customer_Feedback": ["Good", "Avg", "Nice"],
        "Notes": ["-", "-", "-"],

        "Driver_License": ["DL1", "DL2", "DL3"],

        "Daily_Rate": [500, 1500, 800],

        "mileage_flag": ["VALID", "VALID", "VALID"],
        "fuel_flag": ["OK", "LOW", "OK"],
        "overlap_flag": ["NO", "NO", "NO"],

        "License_Valid": [True, True, True],
        "Promo_Valid": [True, False, True],

        "GPS_Lat_Smoothed": [10.05, 20.05, 30.05],
        "GPS_Lon_Smoothed": [30.05, 40.05, 50.05],

        "Expected_Total": [1050.0, 2100.0, 3080.0],
        "GST_Error": [False, False, False],

        "GPS_Lat_Flag": ["OK", "OK", "OK"],
        "GPS_Lon_Flag": ["OK", "OK", "OK"],
        "Drop_Lat_Flag": ["OK", "OK", "OK"],
        "Drop_Lon_Flag": ["OK", "OK", "OK"],
        "Pickup_Lat_Flag": ["OK", "OK", "OK"],
        "Pickup_Lon_Flag": ["OK", "OK", "OK"]
    })




#1 utilization
def test_compute_utilization():

    # create dataframe
    df = sample_df()

    # apply function
    df = compute_utilization(df)

    # check column
    assert "Utilization" in df.columns


#2 revpac
def test_compute_revpac():

    df = sample_df()
    df = compute_revpac(df)

    # check column
    assert "RevPAC" in df.columns


#3 distance and cost
def test_compute_distance_cost():

    df = sample_df()
    df = compute_distance_cost(df)

    # check columns
    assert "Distance_km" in df.columns
    assert "Cost_per_km" in df.columns


#4 idle time
def test_compute_idle_time():

    df = sample_df()

    # utilization required before idle time
    df = compute_utilization(df)

    df = compute_idle_time(df)

    # check column
    assert "Idle_Time" in df.columns


#5 dynamic pricing
def test_dynamic_pricing_features():

    df = sample_df()

    df = dynamic_pricing_features(df)

    # check columns
    assert "Demand" in df.columns
    assert "Lead_Time_Hours" in df.columns
    assert "Month" in df.columns

    # check lead time
    assert df['Lead_Time_Hours'][0] >= 0

    # check month
    assert 1 <= df['Month'][0] <= 12


#6 fuel efficiency
def test_fuel_efficiency():

    df = sample_df()

    df = fuel_efficiency(df)

    # check columns
    assert "Distance_km" in df.columns
    assert "Fuel_Used" in df.columns
    assert "Fuel_Efficiency" in df.columns


#7 damage rate
def test_damage_incidence_rate():

    df = sample_df()

    df = damage_incidence_rate(df)

    # check column
    assert "Damage_Rate" in df.columns


#8 cohort retention
def test_cohort_retention():

    df = sample_df()

    df = cohort_retention(df)

    # check columns
    assert "Cohort_Month" in df.columns
    assert "Activity_Month" in df.columns
    assert "Cohort_Index" in df.columns


#8 nps
def test_nps_rollups():

    df = sample_df()

    df = nps_rollups(df)

    # check columns
    assert "NPS_Category" in df.columns
    assert "NPS_Score" in df.columns


#9 fraud risk
def test_fraud_risk():

    df = sample_df()

    df = fraud_risk(df)

    # check column
    assert "Fraud_Flag" in df.columns


#10 maintenance
def test_maintenance_due_basic():
    df = pd.DataFrame({
        'Odo_Start_km': [0, 0, 0],
        'Odo_End_km': [500, 1200, 600],
        'Fuel_Efficiency': [15, 14, 6],
        'Idle_Time': [100, 150, 200],
        'Pickup_TS': ['2024-01-01', '2024-01-01', '2024-01-01'],
        'Return_TS': ['2024-01-05', '2024-02-15', '2024-01-03']
    })

    # Apply full pipeline
    df = prepare_maintenance_features(df)
    result = maintenance_due_forecast(df)

    expected = [0, 1, 1]

    assert result['Maintenance_Due'].tolist() == expected


#11 overstay
def test_overstay_basic():
    df = pd.DataFrame({
        'Rental_Hours': [10, 25, 30]
    })

    result = overstay_detection(df)

    assert result['Overstay'].tolist() == [0, 1, 1]
    assert result['Overstay_Hours'].tolist() == [0, 1, 6]
    assert result['Penalty_Amount'].tolist() == [0, 500, 500]


#12 punctuality
def test_pickup_punctuality():

    df = sample_df()

    df = pickup_punctuality(df)

    # check column
    assert "Delay" in df.columns


#13 geo hotspots
def test_geo_hotspots():

    df = sample_df()

    df = geo_hotspots(df)

    # check column
    assert "Location" in df.columns


#14 upsell flags
def test_upsell_flags():

    df = sample_df()

    df = upsell_flags(df)

    # check column
    assert "Upsell" in df.columns


#15 cancellation rate
def test_cancellation_rate():

    df = sample_df()

    df = cancellation_rate(df)

    # check column
    assert "Cancelled" in df.columns


#16 driver behavior
def test_driver_behavior():

    df = sample_df()

    df = driver_behavior(df)

    # check column
    assert "Risky_Driver" in df.columns


#17 vehicle mix
def test_vehicle_mix():

    df = sample_df()

    df = vehicle_mix(df)

    # check column
    assert "Vehicle_Count" in df.columns

#18 test fleet health
def test_fleet_health():

    # create dataframe
    df = sample_df()

    # apply function
    df = fleet_health(df)

    # check column exists
    assert "Fleet_Health" in df.columns

    # check valid values
    assert df['Fleet_Health'].isin(["Good", "Bad"]).all()

    # check no null values
    assert df['Fleet_Health'].isnull().sum() == 0


#19 price elasticity
def test_price_elasticity():

    df = sample_df()

    # demand required before elasticity
    df = dynamic_pricing_features(df)

    df = price_elasticity(df)

    # check column
    assert "Price_per_Demand" in df.columns


#20 churn prediction
def test_churn_prediction():

    df = sample_df()

    df = churn_prediction(df)

    # check columns
    assert "Churn_Flag" in df.columns
    assert "Days_Since_Last_Booking" in df.columns