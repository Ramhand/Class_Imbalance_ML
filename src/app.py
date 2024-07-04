import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

try:
    with open('./data/raw/ci.dat', 'rb') as file:
        data = pickle.load(file)
except FileNotFoundError:
    data = 'https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv'
    data = pd.read_csv(data)
    with open('./data/raw/ci.dat', 'wb') as file:
        pickle.dump(data, file)
finally:
    data.drop_duplicates(inplace=True)

cor = data.drop(columns=['STATE_NAME', 'COUNTY_NAME']).corr()
y = data['anycondition_prevalence']
x_possibles = cor.loc[abs(cor['anycondition_prevalence']) > .5]
x_possibles = x_possibles.loc[abs(x_possibles['anycondition_prevalence']) < .73]
x = data[x_possibles.index.to_list()]


# sns.heatmap(pd.concat([x, y], axis=1).corr(), annot=True, cbar=True, fmt='.2f')
# plt.show()

xtr, xte, ytr, yte = train_test_split(x, y, train_size=0.2, random_state=42)
scale = StandardScaler()
scale.fit(xtr)
xtrs = scale.transform(xtr)
xtrs = pd.DataFrame(xtrs, index=xtr.index, columns=xtr.columns)
xtes = scale.transform(xte)
xtes = pd.DataFrame(xtes, index=xte.index, columns=xte.columns)

model = LinearRegression()
model.fit(xtrs, ytr)
predict = model.predict(xtes)
rmodel = Ridge()
rmodel.fit(xtrs, ytr)
rpredict = rmodel.predict(xtes)
lmodel = Lasso()
lmodel.fit(xtrs, ytr)
lpredict = lmodel.predict(xtes)
predicts = [predict, rpredict, lpredict]
for i in predicts:
    print(f'MSE: {mean_squared_error(yte, i)}')
    print(f'RMSE: {np.sqrt(mean_squared_error(yte, i))}')
    print(f'R**2: {r2_score(yte, i)}')

lambda_models = [Ridge(), Lasso()]
alphas = np.logspace(-3, 3, 7)
for i in lambda_models:
    grid_search = GridSearchCV(i, {'alpha':alphas}, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(xtrs, ytr)
    new_model = grid_search.best_estimator_
    pre = new_model.predict(xtes)
    print(f'Best Alpha: {grid_search.best_params_}\nBest RMSE: {np.sqrt(mean_squared_error(yte, pre))}')

lass_list = []
random_ass_alphas = np.linspace(0, 5, 500)
for i in random_ass_alphas:
    lass = Lasso(alpha=i)
    lass.fit(xtrs, ytr)
    pred = lass.predict(xtes)
    lass_list.append(r2_score(yte, pred))
plt.plot(random_ass_alphas, lass_list, color='red')
plt.show()

