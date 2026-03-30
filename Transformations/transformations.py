import pandas as pd
import numpy as np

#1 utilization
def compute_utilization(df):

    # calculating rental hours
    df['Rental_Hours'] = (pd.to_datetime(df['Return_TS']) - pd.to_datetime(df['Pickup_TS'])).dt.total_seconds()/3600

    # calculating utilization
    df['Utilization'] = df['Rental_Hours'] / 24

    return df


#2 revpac
def compute_revpac(df):

    # calculating revenue per available car
    df['RevPAC'] = df['Total_Amount'] / df['Vehicle_ID'].nunique()

    return df


#3 distance and cost
def compute_distance_cost(df):

    # calculating distance
    df['Distance_km'] = df['Odo_End_km'] - df['Odo_Start_km']

    # calculating cost per km
    df['Cost_per_km'] = df['Total_Amount'] / df['Distance_km']

    return df


#4 idle time
def compute_idle_time(df):

    # calculating idle time
    df['Idle_Time'] = 24 - df['Rental_Hours']

    return df


#5 dynamic pricing
def dynamic_pricing_features(df):

    # converting timestamps
    df['Booking_TS'] = pd.to_datetime(df['Booking_TS'])
    df['Pickup_TS'] = pd.to_datetime(df['Pickup_TS'])

    # extracting booking date
    df['booking_date'] = df['Booking_TS'].dt.date

    # calculating demand
    demand = df.groupby(['City','booking_date']).size().reset_index(name='Demand')
    df = df.merge(demand,on=['City','booking_date'],how='left')

    # calculating lead time
    df['Lead_Time_Hours'] = (df['Pickup_TS'] - df['Booking_TS']).dt.total_seconds()/3600

    # extracting month
    df['Month'] = df['Pickup_TS'].dt.month

    return df


#6 fuel efficiency
def fuel_efficiency(df):

    # calculating distance
    df['Distance_km'] = df['Odo_End_km'] - df['Odo_Start_km']
    
    # calculating fuel used
    df['Fuel_Used'] = (1 - df['Fuel_Fraction'])*100
    df['Fuel_Used'] = df['Fuel_Used'].replace(0,0.01)

    # calculating fuel efficiency
    df['Fuel_Efficiency'] = df['Distance_km'] / df['Fuel_Used']

    return df


#7 damage rate
def damage_incidence_rate(df):

    # calculating damage rate
    damage_rate = (df['Damage_Flag'].sum()/len(df))*100

    return round(damage_rate,2)


#8 cohort retention
def cohort_retention(df):

    # converting timestamp
    df['Pickup_TS'] = pd.to_datetime(df['Pickup_TS'])

    # calculating cohort month
    df['Cohort_Month'] = df.groupby('Customer_ID_x')['Pickup_TS'].transform('min').dt.to_period('M')

    # calculating activity month
    df['Activity_Month'] = df['Pickup_TS'].dt.to_period('M')

    # calculating cohort index
    df['Cohort_Index'] = ((df['Activity_Month'].dt.year - df['Cohort_Month'].dt.year)*12 +
                         (df['Activity_Month'].dt.month - df['Cohort_Month'].dt.month))

    return df


#8 nps
def nps_rollups(df):

    # converting feedback to lowercase
    df['Customer_Feedback'] = df['Customer_Feedback'].str.lower()

    # classifying feedback
    df['NPS_Category'] = df['Customer_Feedback'].apply(
        lambda x: 'Promoter' if 'good' in x else ('Passive' if 'smooth' in x else 'Detractor')
    )

    # calculating nps score
    total = len(df)
    promoters = (df['NPS_Category']=='Promoter').sum()
    detractors = (df['NPS_Category']=='Detractor').sum()
    NPS_Score = ((promoters-detractors)/total)*100

    return round(NPS_Score,2)


#9 fraud risk
def fraud_risk(df):

    # detecting odometer rollback
    df['Fraud_Flag'] = np.where(df['Odo_End_km']<df['Odo_Start_km'],1,0)

    return df


#10 maintenance due

def prepare_maintenance_features(df):
    df = df.copy()

    # Convert timestamps
    df['Pickup_TS'] = pd.to_datetime(df['Pickup_TS'])
    df['Return_TS'] = pd.to_datetime(df['Return_TS'])

    # KM used in trip
    df['KM_Since_Service'] = df['Odo_End_km'] - df['Odo_Start_km']

    # Time used (days since trip start)
    df['Days_Since_Trip'] = (df['Return_TS'] - df['Pickup_TS']).dt.days

    return df

def maintenance_due_forecast(df):
    df = df.copy()

    # Thresholds (you can tune these)
    KM_THRESHOLD = 800          # based on your data
    DAYS_THRESHOLD = 30         # service interval
    IDLE_THRESHOLD = 120
    FUEL_THRESHOLD = 12

    df['Maintenance_Due'] = (
        (df['KM_Since_Service'] > KM_THRESHOLD) |
        (df['Days_Since_Trip'] > DAYS_THRESHOLD) |
        (df['Fuel_Efficiency'] < FUEL_THRESHOLD) |
        (df['Idle_Time'] > IDLE_THRESHOLD)
    ).astype(int)

    return df


#11 overstay detection
def overstay_detection(df):

    # checking rental over 24 hours
    df['Overstay'] = np.where(df['Rental_Hours'] > 24, 1, 0)

    # penalty based on how many hours overdue
    df['Overstay_Hours'] = np.where(df['Rental_Hours'] > 24, df['Rental_Hours'] - 24, 0)

    df['Penalty_Amount'] = np.where(df['Overstay_Hours'] == 0, 0,
                           np.where(df['Overstay_Hours'] <= 6,   500,
                           np.where(df['Overstay_Hours'] <= 12,  1000,
                           np.where(df['Overstay_Hours'] <= 24,  1500,
                                                                 2000))))

    return df

#12 punctuality
def pickup_punctuality(df):

    # calculating delay
    df['Delay'] = (pd.to_datetime(df['Pickup_TS']) - pd.to_datetime(df['Booking_TS'])).dt.total_seconds()/3600

    return df


#13 geo hotspots
def geo_hotspots(df):

    # assigning location
    df['Location'] = df['City']

    return df


#14 upsell flags
def upsell_flags(df):

    # checking promo usage
    df['Upsell'] = np.where(df['Promo_Valid']==True,1,0)

    return df


#15 cancellation rate
def cancellation_rate(df):

    # marking cancellations
    df['Cancelled'] = np.where(df['Total_Amount']==0,1,0)

    return df


#16 driver behavior
def driver_behavior(df):

    # identifying risky driving
    df['Risky_Driver'] = np.where(df['Harsh_Events']>2,1,0)

    return df


#17 vehicle mix
def vehicle_mix(df):

    # counting vehicles by class
    df['Vehicle_Count'] = df['Vehicle_Class'].map(df['Vehicle_Class'].value_counts())

    return df


#18 price elasticity
def price_elasticity(df):

    # calculating price per demand
    df['Price_per_Demand'] = df['Total_Amount']/df['Demand']

    return df


#19 fleet health
def fleet_health(df):

    # evaluating fleet condition
    df['Fleet_Health'] = np.where((df['Damage_Flag']==0) & (df['Harsh_Events']<2),'Good','Bad')

    return df

#20 churn likelihood
def churn_prediction(df):

    # converting timestamp
    df['Pickup_TS'] = pd.to_datetime(df['Pickup_TS'])

    # latest booking per customer
    last_booking = df.groupby('Customer_ID_x')['Pickup_TS'].transform('max')

    # calculating days since last booking
    df['Days_Since_Last_Booking'] = (df['Pickup_TS'].max() - last_booking).dt.days

    # identifying churn (no booking for long time)
    df['Churn_Flag'] = np.where(df['Days_Since_Last_Booking'] > 30, 1, 0)

    return df
