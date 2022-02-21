#  Predict how well can 21 days lockdown perform in containing spread of COVID19 Virus.

#   Import the Data from csv file, file name : Covid_19_india.csv
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

raw_data = pd.read_csv("covid_19_india.csv")

#   Converting raw data into required data
def show_data():
    required_data = raw_data[["Date", "State/UnionTerritory", "Cured", "Deaths", "Confirmed"]]
    required_data['Date'] = pd.to_datetime(required_data['Date'])

    #   Inserting column with name lockdown giving Categorical attribute

    required_data['Lockdown'] = 'No';
    required_data['Lockdown'][
        (required_data['Date'] >= dt.datetime(2020, 3, 24)) & (required_data['Date'] <= dt.datetime(2020, 5, 31))] = 'Yes'

    m = max(required_data['Confirmed'])  # correspond to column 18094

    st.subheader('calculating percent wrt to max confirMed case')
    required_data['percent_suffer'] = (required_data['Confirmed'] / m) * 100
    required_data.rename({"State/UnionTerritory": "State"}, axis=1, inplace=True)
    st.dataframe(required_data.head(20))

    df = required_data

    st.subheader('EDA on data')

    st.write(df.info())
    st.write(df.describe())

    #    Graphs
    st.subheader('Confirmed Cases vs Cases Cured')

    plota = sns.jointplot(x="Confirmed", y="Cured", data=df)
    st.pyplot(plota)

    st.subheader('Confirmed Cases vs Deaths')

    plotb = sns.jointplot(x="Confirmed", y="Deaths", data=df)
    st.pyplot(plotb)

    st.subheader('Graph of categorical attribute "Lockdown"')
    fig = plt.figure(figsize=(10, 20))
    sns.countplot(x="Lockdown", data=df)          # print
    # sns.set(rc={'figure.figsize': (10, 20)})
    st.pyplot(fig)

    #   Finding total number of Deaths in each States

    uniqueState = df['State'].unique()

    d = np.array([])
    for state in uniqueState:
        total = df.loc[df["State"] == state, 'Deaths'].sum()
        d = np.append(d, total)

    st.subheader('Ploting graph of States vs Total Deaths')


    fig = plt.figure(figsize=(15, 9))
    X_axis = np.arange(len(uniqueState))
    plt.bar(X_axis, d, 0.5)
    plt.xticks(X_axis, uniqueState, rotation='vertical')
    plt.xlabel("States")
    plt.ylabel("Deaths")
    plt.title("Total Deaths in each States")
    # plt.show()
    st.pyplot(fig)
    st.subheader('Classifing the dataset as per presence of Lockdown (Yes/No)')

    yes = df[(df.Lockdown == "Yes")]
    # print("Dataset with lockdown")
    st.write("Dataset with lockdown")
    st.dataframe(yes)
    # print(yes)
    no = df[(df.Lockdown == "No")]
    # print("Dataset with Unlock")
    st.write("Dataset with Unlock")
    # print(no)
    st.dataframe(no)
    st.subheader('Making dataset only for number of Deaths in Lockdown And in Unlock')
    newdf = pd.DataFrame(list(zip(yes.Deaths, no.Deaths)), columns=['In_Lockdown', 'In_Unlock'])
    # print("Dataset with number of deaths in lockdown and in unlock ")
    st.subheader("Dataset with number of deaths in lockdown and in unlock ")    # print
    st.dataframe(newdf.head(10))    # print
    # print(newdf.head(10))

    st.subheader('Total Number of deaths in Lockdown and in Unlock')
    deathsInLockdown = newdf["In_Lockdown"].values
    totaldeathsInLockdown = np.sum(deathsInLockdown)
    st.write("Total deaths In Lockdown ", totaldeathsInLockdown)   # print

    deathInUnlock = newdf["In_Unlock"].values
    totaldeathInUnlock = np.sum(deathInUnlock)
    st.write("Total death In Unlock", totaldeathInUnlock)  # print

    st.subheader('Bar Graph of total Number of deaths in Lockdown and in Unlock')
    st.write("total Number of deaths in Lockdown and in Unlock")   # print
    fig = plt.figure(figsize=(10, 9))
    plt.bar("totaldeathsInLockdown", totaldeathsInLockdown)
    plt.bar("totaldeathInUnlock", totaldeathInUnlock)
    plt.title("Comparision between total number of deaths India  lockdown and in unlock ")
    # plt.show()  # print
    st.pyplot(fig)

    st.subheader('Taking Data for only one State Maharashtra')
    st.write("Dataset containing only data of Maharashtra")    # print
    Mahadf = df.loc[(df.State == 'Maharashtra')]
    st.dataframe(Mahadf.head(10))  # print

    st.subheader('Garph of Days of lockdown and Unlock in Maharashtra')
    st.write("Days of lockdown and unlock in Maharashtra") # print
    fig = plt.figure(figsize=(10, 20))
    sns.countplot(x="Lockdown", data=Mahadf)
    # sns.set(rc={'figure.figsize': (10, 20)})
    # plt.show()  # print
    st.pyplot(fig)

    #   Scatter plot of Deaths in Lockdown and in Unlock
    st.subheader("Scatter plot of Deaths in Lockdown and in Unlock")   # print
    fig = plt.figure(figsize=(10, 10))
    sns.catplot(x="Lockdown", y="Deaths", data=Mahadf)
    st.pyplot(fig)
    # plt.show()  # print

    # plotting Death rate in Maharashtra
    st.subheader("Death rate in Maharashtra")  # print
    fig, ax = plt.subplots(figsize=(15, 7))
    x_values = Mahadf["Date"].values
    y_values = Mahadf["Deaths"].values
    plt.plot(x_values, y_values)    # print
    plt.title("Death rate in Maharashtra", fontsize=14, fontweight='bold')
    ax.set_xlabel("Dates", fontsize=14)
    ax.set_ylabel("Deaths", fontsize=14)
    # plt.show()  # print
    st.pyplot(fig)

    #   classifing the Maharashtra datset into two dataset on the base of lockdown
    MahaWithLockdown = Mahadf.loc[(df.Lockdown == 'Yes')]
    MahaWithUnlock = Mahadf.loc[(df.Lockdown == 'No')]

    st.subheader("Death rate in Maharashtra in Lockdown")  # print

    datesofLockdown = MahaWithLockdown["Date"]
    deathsInLockdown = MahaWithLockdown["Deaths"]
    fig, ax = plt.subplots(figsize=(15, 7))
    date = [pd.to_datetime(d) for d in datesofLockdown]
    plt.scatter(datesofLockdown, deathsInLockdown, s=5, c='red')
    plt.title("Death rate in Maharashtra (Lockdown)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Dates", fontsize=14)
    ax.set_ylabel("Deaths", fontsize=14)
    # plt.show()  # print
    st.pyplot(fig)

    st.subheader("Death rate in Maharashtra in Unlock")    # print
    datesofUnlock = MahaWithUnlock["Date"]
    deathsInUnlock = MahaWithUnlock["Deaths"]
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.scatter(datesofUnlock, deathsInUnlock, s=5, c='blue')
    plt.title("Death rate in Maharashtra (Unlock)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Dates", fontsize=14)
    ax.set_ylabel("Deaths", fontsize=14)
    # plt.show()  # print
    st.pyplot(fig)

    #   Calculating total number of Deaths for the number of days
    # In Lockdown
    xarray = np.array([])  # contain number of days
    yarray = np.array([])  # contain number of deaths
    data = np.array([])
    for i in range(MahaWithLockdown.shape[0]):
        dataset = MahaWithLockdown.iloc[:i]
        total = dataset.Deaths.sum()
        xarray = np.append(xarray, i)
        yarray = np.append(yarray, total)

    # Plotting the graph between number of days of lockdown vs deaths
    st.subheader("Graph between number of days of lockdown vs deaths") # print
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.scatter(xarray, yarray, s=5, c='blue')
    plt.title("Number of days of lockdown vs Deaths ", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Days (Lockdown)", fontsize=14)
    ax.set_ylabel("Deaths", fontsize=14)
    # plt.show()  # print
    st.pyplot(fig)

    #   we use scipy.stats.linregress for linear regression in our model

    st.write("We use scipy.stats.linregress for linear regression in our model")
    st.subheader("Graph of regression line (in lockdown)")
    from scipy import stats

    fig, ax = plt.subplots(figsize=(15, 7))

    slope, intercept, r, p, std_err = stats.linregress(xarray, yarray)


    def myfunc(xarray):
        return slope * xarray + intercept


    mymodel = list(map(myfunc, xarray))

    plt.scatter(xarray, yarray, color='blue')

    plt.plot(xarray, mymodel)

    plt.title('Regression Line', fontsize=15)
    plt.xlabel('Days of Lockdown', fontsize=14)
    plt.ylabel('Deaths', fontsize=14)
    # plt.show()  # print
    st.pyplot(fig)

    st.write("Predicting the number of deaths int 100 days of lockdown")
    st.write(int(myfunc(100)))

    #   Calculating total number of Deaths for the number of days
    #   in Unlock

    xarray = np.array([])  # contain number of days
    yarray = np.array([])  # contain number of deaths
    data = np.array([])
    for i in range(MahaWithUnlock.shape[0]):
        dataset = MahaWithUnlock.iloc[:i]
        total = dataset.Deaths.sum()
        xarray = np.append(xarray, i)
        yarray = np.append(yarray, total)

    fig, ax = plt.subplots(figsize=(15, 7))
    print("Graph Number of days of Unlock vs Deaths")
    plt.scatter(xarray, yarray, s=5, c='blue')
    plt.title("Number of days of Unlock vs Deaths ", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Days (Unlock)", fontsize=14)
    ax.set_ylabel("Deaths", fontsize=14)
    # plt.show()  # print
    st.pyplot(fig)

    st.subheader("Graph of regression line (in Unlock) ")

    fig, ax = plt.subplots(figsize=(15, 7))

    slope, intercept, r, p, std_err = stats.linregress(xarray, yarray)


    def myfunc(xarray):
        return slope * xarray + intercept


    mymodel = list(map(myfunc, xarray))

    plt.scatter(xarray, yarray, color='blue')

    plt.plot(xarray, mymodel)

    plt.title('Regression Line', fontsize=15)
    plt.xlabel('Days of Unlock', fontsize=14)
    plt.ylabel('Deaths', fontsize=14)
    # plt.show()  # print
    st.pyplot(fig)

    st.write("Predicting the number of deaths in 100 days of lockdown")
    st.write(int(myfunc(100)))    # print

    st.write("At last we predict the number of deaths in 100 days of lockdown and 100 days of unlock ")
    st.write("Manually we can say that the lockdown is successful to prevent the contamination of COVID19 virus")
