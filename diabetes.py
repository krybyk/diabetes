import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template, request
import joblib
import streamlit as st

st.title("Diyabet Tahmin")



sayısı = st.slider('Hamile kalma sayınız', 0,15, step = 1)
glukoz = st.slider('Glukoz miktarınızı giriniz', 0,200, step = 1)
basıncı= st.slider('Nabzınızı giriniz', 40,120, step = 1)
kalınlığı = st.slider('Deri kalınlığınızı giriniz', 0,90, step = 1)
insülin = st.slider('İnsülin miktarınızı giriniz', 0,850, step = 1)
endeks = st.slider('Vücut kitle endeksinizi giriniz', 0,70, step = 1)
insan = st.slider("Ailedeki diyabetli insan sayısı", 0,20, step= 1)
yaş = st.slider("yaş", 0, 100, step= 1)


kullanıcı = [sayısı, glukoz, basıncı, kalınlığı, insülin, endeks, insan, yaş]


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Datasets/diabetes.csv")
df.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]


    return cat_cols, cat_but_car, num_cols

grab_col_names(df)
cat_cols, cat_but_car, num_cols = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()




def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

    print("#####################################")



corr = df[num_cols].corr()




def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=False)

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

#Outcome hariç hepsinde aykırı değerler var

#Değerleri low yada up değerine baskılıyorum
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_

df.isnull().values.any()
#Eksik değer yok

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
dff.head()

X = dff.drop(["Outcome"], axis=1)
y = dff["Outcome"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,shuffle=True,
                                               random_state = 0 )

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k = 3
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred)

sample_data = np.array(kullanıcı)
sample_data = sample_data.reshape(1,-1)
sample_data = scaler.transform(sample_data)
prediction = knn_model.predict(sample_data)

if st.button('Hesapla'):
    if prediction == 0:
        st.success("Diyabetsiniz.")
    else:
        st.success("Diyabet değilsiniz.")