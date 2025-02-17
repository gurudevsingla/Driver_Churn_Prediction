# Problem Statement

Recruiting and retaining drivers is seen by industry watchers as a tough battle for Ola. Churn among drivers is high and it's very easy for drivers to stop working for the service on the fly or jump to Uber depending on the rates.


As the companies get bigger, the high churn could become a bigger problem. To find new drivers, Ola is casting a wide net, including people who don't have cars for jobs. But this acquisition is really costly. Losing drivers frequently impacts the morale of the organization and acquiring new drivers is more expensive than retaining existing ones.


You are working as a data scientist with the Analytics Department of Ola, focused on driver team attrition. You are provided with the monthly information for a segment of drivers for 2019 and 2020 and tasked to predict whether a driver will be leaving the company or not based on their attributes like

- Demographics (city, age, gender etc.)
- Tenure information (joining date, Last Date)
- Historical data regarding the performance of the driver (Quarterly rating, Monthly business acquired, grade, Income)


Dataset Link - https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/002/492/original/ola_driver_scaler.csv

# Approach

- Data Preprocessing and Feature Engineering
  - Data is collected on multiple dates from a driver so first aggregating it to get the data on Driver ID level.
  - Created Binary Features like Quaterly Rating Has Increased or not, Monthly income for the driver has increased or not.
  - Made different infrences which are present in ipynb.
- Modelling
  - Used Logistic Regression, Random Forest, Gradient Boosted Decision Trees
  - Best F1-Score and Area Under Curve recieved using Random Forest i.e. 0.87 and 0.75

# Deployed on Streamlit

Link - https://driver-churn-prediction.streamlit.app/
