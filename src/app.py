# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from statsmodels.api import OLS
import pickle

# Load data
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv',header=0, sep=',')

# Selecting the variables and the target 
X = df_raw.drop(['CNTY_FIPS','fips','Active Physicians per 100000 Population 2018 (AAMC)','Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)', 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)', 'Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)','Active General Surgeons per 100000 Population 2018 (AAMC)','Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)','Total nurse practitioners (2019)','Total physician assistants (2019)','Total physician assistants (2019)','Total Hospitals (2019)','Internal Medicine Primary Care (2019)','Family Medicine/General Practice Primary Care (2019)','STATE_NAME','COUNTY_NAME','ICU Beds_x','Total Specialist Physicians (2019)'], axis=1)
y = df_raw['ICU Beds_x']

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeling with Lasso
pipeline = make_pipeline(StandardScaler(), Lasso(alpha=1))
pipeline.fit(X_train, y_train)

# Guardar el modelo
filename = '../models/modelo_reg_regresion_lineal.sav'
pickle.dump(pipeline, open(filename, 'wb'))

loc = [i for i, e in enumerate(pipeline[1].coef_) if e != 0]
col = X.columns
col[loc]

spector_data = X_train[["30-39 y/o % of total pop", "40-49 y/o % of total pop",
       'Black-alone pop', '% Asian-alone', 'GQ_ESTIMATES_2018', 'R_birth_2018',
       "Bachelor's degree or higher 2014-18",
       'Percent of adults with less than a high school diploma 2014-18',
       'Percent of adults with a high school diploma only 2014-18',
       "Percent of adults with a bachelor's degree or higher 2014-18",
       'MEDHHINC_2018', 'CI90LBINC_2018', 'Unemployment_rate_2018',
       'COPD_number']]
       
x_spector_data = sm.add_constant(spector_data)

# Fit and summarize OLS model
mod = sm.OLS(y_train, x_spector_data)
res = mod.fit()

print('R2 train', pipeline.score(X_train, y_train)*100)
print('R2 test', pipeline.score(X_test, y_test)*100)
print(res.summary())

