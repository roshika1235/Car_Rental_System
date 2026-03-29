import pandas as pd
import re
import numpy as np


# ---------- LOAD DATASET ----------
def load_dataset():
    # Read dataset from given path
    df = pd.read_csv(r"C:\Users\Administrator\OneDrive\Desktop\Case_Study\car_rental.csv")
    return df


# ---------- 1. CLEAN VEHICLE ----------
def clean_vehicle_id(df):
    # Convert to uppercase, remove spaces, replace inner spaces with "-"
    df['Vehicle_ID'] = df['Vehicle_ID'].str.upper().str.strip().str.replace(' ', '-')
    return df

# ----------2. FIX TIMESTAMP ----------
def fix_timestamp(col):

    # converting to datetime safely
    return pd.to_datetime(col, errors='coerce')


# ---------- NORMALIZE TIME ----------
def normalize_time(ts):

    # skip null values
    if pd.isna(ts):
        return ts

    # extract components
    sec = ts.second
    minute = ts.minute
    hour = ts.hour

    # fix seconds overflow
    minute += sec // 60
    sec = sec % 60

    # fix minutes overflow
    hour += minute // 60
    minute = minute % 60

    # return corrected timestamp
    return ts.replace(hour=hour % 24, minute=minute, second=sec)


# ---------- APPLY TIMESTAMP CLEANING ----------
def clean_timestamps(df):

    # apply safe conversion
    df["Booking_TS"] = fix_timestamp(df["Booking_TS"])
    df["Pickup_TS"] = fix_timestamp(df["Pickup_TS"])
    df["Return_TS"] = fix_timestamp(df["Return_TS"])

    # apply normalization
    df["Booking_TS"] = df["Booking_TS"].apply(normalize_time)
    df["Pickup_TS"] = df["Pickup_TS"].apply(normalize_time)
    df["Return_TS"] = df["Return_TS"].apply(normalize_time)

    # fill missing pickup with booking
    df["Pickup_TS"] = df["Pickup_TS"].fillna(df["Booking_TS"])

    # drop remaining null pickup
    df = df.dropna(subset=["Pickup_TS"])

    return df

   
# ---------- 3. CLEAN ODO ----------
def clean_odo_start(df):
    # Convert to string and remove unwanted characters
    df['Odo_Start_km'] = (
        df['Odo_Start_km']
        .astype(str)                       # ensure string type
        .str.replace('km', '', regex=False)  # remove "km"
        .str.replace(',', '', regex=False)   # remove commas
        .str.strip()                        # remove spaces
    )
    # Convert to numeric, invalid → NaN
    df['Odo_Start_km'] = pd.to_numeric(df['Odo_Start_km'], errors='coerce')
    return df


def clean_odo_end(df):
    # Same cleaning for Odo_End
    df['Odo_End_km'] = (
        df['Odo_End_km']
        .astype(str)
        .str.replace('km', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df['Odo_End_km'] = pd.to_numeric(df['Odo_End_km'], errors='coerce')
    return df


# ---------- 3.1 PREPARE ODO (RENAMING) ----------
def prepare_odo(df):
    # Rename columns if needed
    df.rename(columns={
        'Odo_Start': 'Odo_Start_km',
        'Odo_End': 'Odo_End_km'
    }, inplace=True)

    # Apply cleaning
    df = clean_odo_start(df)
    df = clean_odo_end(df)

    return df


# ---------- 4. FUEL ----------
def clean_fuel_level(df):

    # Prevent duplicate column issue
    if 'Fuel_Fraction' in df.columns:
        df = df.drop(columns=['Fuel_Fraction'])

    # Remove % and convert to float
    df['Fuel_Level'] = df['Fuel_Level'].str.replace('%', '').str.strip().astype(float)

    # Convert >1 values to fraction (e.g., 50 → 0.5)
    df['Fuel_Level'] = df['Fuel_Level'].apply(lambda x: x/100 if x > 1 else x)

    # Rename column
    df.rename(columns={'Fuel_Level': 'Fuel_Fraction'}, inplace=True)

    return df


# ----------5. RATE ----------
def clean_rate(df):

    # Remove ₹ symbol
    df['Rate'] = df['Rate'].str.replace('₹', '')

    # Remove "/day"
    df['Rate'] = df['Rate'].str.replace('/day','')

    # Remove commas
    df['Rate'] = df['Rate'].str.replace(',', '')

    # Remove spaces
    df['Rate'] = df['Rate'].str.strip()

    # Convert to integer → Daily_Rate
    df['Daily_Rate'] = df['Rate'].astype(int)

    # Drop old column
    df = df.drop(columns=['Rate'])

    return df


# ----------6. CITY ----------

def clean_city(df):
    # Mapping incorrect -> correct city names
    dict_change_city = {
        'delhi': 'Delhi',
        'kolkata': 'Kolkata',
        'chennai': 'Chennai',
        'mumbai': 'Mumbai',
        'bangalore': 'Bengaluru',
        'bengaluru': 'Bengaluru',
        'blr': 'Bengaluru',
        'hyd': 'Hyderabad',
        'hyderabad': 'Hyderabad'
    }

    # Convert to lowercase and replace
    df['City'] = df['City'].str.lower().replace(dict_change_city)
    return df


# ----------7. DUPLICATES ----------
def remove_duplicate_reservation(df):
    # Sort by pickup time
    df = df.sort_values(by='Pickup_TS')

    # Remove duplicate Reservation_ID (keep first)
    df = df.drop_duplicates(subset='Reservation_ID', keep='first')
    return df


# ----------8. PICKUP_RETURN RULE VALIDATION ----------
def validate_trip_time(df):
    """
    Validate trip timestamps safely (handles dirty/mixed formats)
    """

    # Safe datetime conversion (NO crash)
    df['Pickup_TS'] = pd.to_datetime(df['Pickup_TS'], errors='coerce')
    df['Return_TS'] = pd.to_datetime(df['Return_TS'], errors='coerce')

    # Remove invalid datetime rows
    df = df.dropna(subset=['Pickup_TS', 'Return_TS'])

    #  Apply rule
    df = df[df['Return_TS'] >= df['Pickup_TS']]

    # Reset index (prevents pytest KeyError)
    df = df.reset_index(drop=True)

    return df


# ----------9. PAYMENT ----------
def clean_payment(df):
    # Convert payment mode to uppercase
    df['Payment'] = df['Payment'].str.upper()
    return df


# ----------10. MILEAGE ----------
def mileage_sanity_check(df):

    def flag_row(row):
        start, end = row['Odo_Start_km'], row['Odo_End_km']

        # Missing values
        if pd.isnull(start) or pd.isnull(end):
            return "INVALID"
        # Odo rollback
        elif end < start:
            return "ERROR"
        # No movement
        elif end == start:
            return "ZERO"
        # Too high distance
        elif (end - start) > 800:
            return "HIGH"
        # Normal case
        else:
            return "VALID"

    # Apply row-wise
    df['mileage_flag'] = df.apply(flag_row, axis=1)
    return df


# ----------11. FUEL SANITY ----------
def fuel_sanity_check(df):

    fuel_flags = []

    for i in range(len(df)):
        # Read values
        fuel = df.iloc[i]['Fuel_Fraction']
        start = df.iloc[i]['Odo_Start_km']
        end = df.iloc[i]['Odo_End_km']

        # Check missing
        if pd.isnull(fuel) or pd.isnull(start) or pd.isnull(end):
            fuel_flags.append("INVALID")
            continue

        # Calculate distance
        distance = end - start

        # Apply rules
        if distance == 0 and fuel < 0.2:
            fuel_flags.append("LOW_FUEL_NO_USAGE")
        elif distance == 0 and fuel > 0.8:
            fuel_flags.append("POSSIBLE_REFUEL")
        elif distance > 0 and fuel < 0.2:
            fuel_flags.append("HIGH_CONSUMPTION")
        else:
            fuel_flags.append("NORMAL")

    # Add column
    df['fuel_flag'] = fuel_flags
    return df


# ----------12. VEHICLE AVAILABILITY OVERLAP ----------
def trip_overlap_check(df):

    # Sort by vehicle and time
    df = df.sort_values(by=['Vehicle_ID', 'Pickup_TS'])

    overlap_flags = []

    for i in range(len(df)):

        # First row case
        if i == 0:
            overlap_flags.append("NO_PREVIOUS")
            continue

        curr_vehicle = df.iloc[i]['Vehicle_ID']
        prev_vehicle = df.iloc[i-1]['Vehicle_ID']

        curr_start = df.iloc[i]['Pickup_TS']
        prev_end = df.iloc[i-1]['Return_TS']

        # Same vehicle check
        if curr_vehicle == prev_vehicle:
            if curr_start < prev_end:
                overlap_flags.append("OVERLAP")
            else:
                overlap_flags.append("NO_OVERLAP")
        else:
            overlap_flags.append("NEW_VEHICLE")

    df['overlap_flag'] = overlap_flags
    return df


# ---------- 13. DAMAGE ANALYSIS ----------
def damage_analysis(df):

    # Filter damaged trips
    damage_cases = df[df["Damage_Flag"] == 1]

    # Calculate percentage distribution
    damage_rate = df["Damage_Flag"].value_counts(normalize=True) * 100

    return damage_cases, damage_rate


# ----------14. DRIVER LICENSE CLEANING ----------
def clean_driver_license(df):

    # Fill missing values
    df["Driver_License"] = df["Driver_License"].fillna("UNKNOWN")

    # Replace invalid entries
    df.loc[df["Driver_License"] == "INVALID", "Driver_License"] = "UNKNOWN"

    # Validate format DL12345
    df["License_Valid"] = df["Driver_License"].str.match(r"^DL\d{5}$")

    # Mask license
    def mask_license(x):
        if x == "UNKNOWN":
            return x
        return x[:2] + "***" + x[-2:]

    df["Driver_License"] = df["Driver_License"].apply(mask_license)

    return df


# ----------15. PROMO CODE CLEANING ----------
def clean_promo_code(df):

    # Fill missing values
    df["Promo_Code"] = df["Promo_Code"].fillna("NO_PROMO")

    # Clean text
    df["Promo_Code"] = df["Promo_Code"].astype(str).str.strip()

    # Validate format
    df["Promo_Valid"] = df["Promo_Code"].str.match(r'^[A-Z]+[0-9]+$')

    return df


# ---------- 16. GPS JITTER SMOOTHING ----------
def smooth_gps(df):

    # Round GPS values
    df["GPS_Lat_Smoothed"] = df["GPS_Lat"].round(4)
    df["GPS_Lon_Smoothed"] = df["GPS_Lon"].round(4)

    return df


# ----------17. HARSH EVENTS ----------
def normalize_harsh_events(df):

    # Fill missing
    df['Harsh_Events'] = df['Harsh_Events'].fillna(0)

    # Convert to binary
    df['Harsh_Events'] = df['Harsh_Events'].apply(lambda x: 1 if x > 0 else 0)

    return df


# ----------18. PII REDACTION ----------
def redact_pii(df):

    def mask_pii(text):
        if pd.isnull(text):
            return text

        text = str(text)

        # Mask phone numbers
        text = re.sub(r'\d{10}', '**********', text)

        # Mask emails
        text = re.sub(r'\S+@\S+', '*****@*****', text)

        return text

    df['Notes'] = df['Notes'].apply(mask_pii)
    df['Customer_Feedback'] = df['Customer_Feedback'].apply(mask_pii)

    return df


# ----------19. RATE PLAN ----------
def prepare_rate_plan_data(df):

    # Clean rate plan
    df['Rate_Plan'] = df['Rate_Plan'].astype(str).str.strip().str.lower()

    df['Rate_Plan'] = df['Rate_Plan'].replace({
        'std': 'standard',
        'prem': 'premium',
        'eco': 'economy'
    })

    # Convert rate
    df['Daily_Rate'] = pd.to_numeric(df['Daily_Rate'], errors='coerce')

    # Mapping
    rate_mapping = {
        'standard': 1500,
        'premium': 2500,
        'economy': 1000
    }

    # Fill missing
    df['Daily_Rate'] = df['Daily_Rate'].fillna(df['Rate_Plan'].map(rate_mapping))

    return df


# ----------20. GST VALIDATION ----------
def validate_gst(df):

    # Convert to numeric
    df['Daily_Rate'] = pd.to_numeric(df['Daily_Rate'], errors='coerce')
    df['GST_Amount'] = pd.to_numeric(df['GST_Amount'], errors='coerce')
    df['Total_Amount'] = pd.to_numeric(df['Total_Amount'], errors='coerce')

    # Fill missing GST
    mean_val = df['GST_Amount'].mean()
    df['GST_Amount'] = df['GST_Amount'].fillna(mean_val).round(0)

    # Calculate expected total
    df['Expected_Total'] = df['Daily_Rate'] + df['GST_Amount']

    # Flag mismatch
    df['GST_Error'] = df['Total_Amount'] != df['Expected_Total']

    return df