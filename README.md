# 🚗 Car Rental System —  Case Study

A comprehensive data engineering project focused on cleaning, transforming, and analyzing car rental operations data across multiple business domains including reservations, payments, telematics, and customer feedback.

---

## 📁 Project Structure
```
Car_Rental_System/
│
├── cleaning/
│   ├── cleaner.py               # Data cleaning logic and functions
│   └── __init__.py
│
├── Transformations/
│   ├── transformations.py       # Data transformation pipelines
│   └── __init__.py
│
├── testing/
│   ├── test_cleaning.py         # Unit tests for cleaning module
│   └── test_transformation.py  # Unit tests for transformation module
│
├── datasets/                    # Raw input datasets
│   ├── Reservations.csv
│   ├── Payments.csv
│   ├── Telematics.csv
│   ├── Maintenance_Log.csv
│   ├── Customer_Feedback.csv
│   └── feedback.csv
│
├── notebook/
│   └── Data_Merging.ipynb       # Data merging and integration notebook
│
├── Data_Cleaning.ipynb          # Notebook: cleaning walkthrough
├── Data_Transformation.ipynb    # Notebook: transformation walkthrough
├── task2.ipynb                  # Task 2 analysis notebook
│
├── car_rental_cleaned_dataset.csv     # Output: cleaned data
├── car_rental_transformed.csv         # Output: transformed data
├── car_rental.csv                     # Base dataset
└── requirements.txt                   # Python dependencies
```

---

## 📌 Project Overview

This case study simulates a real-world data pipeline for a car rental business. The project covers:

- **Data Ingestion** — Loading raw CSVs across multiple operational domains
- **Data Cleaning** — Handling missing values, duplicates, type conversions, and inconsistencies
- **Data Transformation** — Feature engineering, normalization, and aggregations
- **Data Merging** — Joining datasets across reservations, payments, and telematics
- **Testing** — Automated unit tests to validate cleaning and transformation logic

---

## 📊 Datasets

| Dataset | Description |
|---|---|
| `Reservations.csv` | Booking records including dates, vehicle, and customer info |
| `Payments.csv` | Payment transactions linked to reservations |
| `Telematics.csv` | GPS and vehicle usage telemetry data |
| `Maintenance_Log.csv` | Vehicle maintenance and service history |
| `Customer_Feedback.csv` | Customer ratings and review data |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Car_Rental_System.git
cd Car_Rental_System
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the cleaning pipeline
```bash
python cleaning/cleaner.py
```

### Run the transformation pipeline
```bash
python Transformations/transformations.py
```

### Run tests
```bash
pytest testing/
```

### Explore notebooks
Open any `.ipynb` file in Jupyter or VS Code to walk through the analysis step by step.

---

## 🧪 Testing

Unit tests are written using `pytest` and cover:
- Null value handling
- Data type validation
- Transformation correctness
- Edge cases in cleaning logic
```bash
pytest testing/ -v
```

---

## 🛠️ Tech Stack

- **Python 3.13.12**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Pytest** — unit testing
- **Jupyter Notebook** — exploratory analysis

---

## 👤 Author

## 👥 Authors

**Roshika Challa**
[GitHub](https://github.com/roshika1235) | [LinkedIn](https://www.linkedin.com/in/roshika-challa-307575259-bvrit-hyderabad-college-of-engineering-for-women/)

**Bathula Padmasri**
[GitHub]((https://github.com/padmasri2005)) | [LinkedIn](https://linkedin.com/in/?)

**Sanjana Gandla**
[GitHub](https://github.com/Sanjana6653) | [LinkedIn](https://linkedin.com/in/?)

**Siri Chandana Macha**
[GitHub](https://github.com/siri-chandana-macha) | [LinkedIn](https://linkedin.com/in/?)

**Sharanya Lankela**
[GitHub](https://github.com/?) | [LinkedIn](https://linkedin.com/in/?)
