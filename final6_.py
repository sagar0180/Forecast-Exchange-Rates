#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model

st.title('Predicting Exchange Rates USD VS INR')
st.write('  note: exchange rates are expressed as INR per USD')

mod_lstm = load_model( r'C:\Users\MMM-SM\21Pypractice\project\FinalSubmission\lstm_model.h5',)


df = pd.read_excel(r'C:\Users\MMM-SM\21Pypractice\project\DEXINUS.xls', skiprows=10,index_col=[0],parse_dates=True)
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(1,1,figsize=(15,3))
ax.plot(df)
plt.xticks(rotation='vertical')
st.write(fig)
#st.line_chart(df)
df = df.iloc[12000:,]

#Fill missing values using interpolation
df['DEXINUS'].interpolate(method='linear',limit_direction='forward',inplace=True)


# Apply minmaxscaler on the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df)
data1 = scaler.transform(df)


# get an input array for lstm model
x_input=np.array(data1[-177:]).reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


st.title(">Welcome to the future exchange rates<")
name = st.text_input("Enter your name here")

day = st.text_input(" how many days of prediction you want ")


# demonstrate prediction for next n days
#day = 30

if day:
    c = int(day)
    st.write(f'Wait while we make your prediction for next {c} days......')
    len(temp_input)
    lst_output=[]
    n_steps=176
    i=0    
    while(i<c):
        if(len(temp_input)>1):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            # print("{} day input {}".format(i,x_input))
            # x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, 1, n_steps))
            # print(x_input)
            yhat = mod_lstm.predict(x_input, verbose=0)
            # print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape(1,1,n_steps,)
            print(x_input)
            yhat = mod_lstm.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    result = scaler.inverse_transform(lst_output)
    future_dates = pd.date_range(start = '27-06-2021', periods=c , freq = 'B').format('yyyy-mmm-dd')[1:]
    future_dates = pd.Series(future_dates,name='Business_Days',) 

    results = pd.DataFrame(result,columns=['Exch_Rate'])

    final_df=pd.concat([future_dates,results],axis=1)
    
    
    #st.button("=> Click to Predict")):
    st.subheader(f'Predicted exchange rate for next {c} days') 
    st.dataframe(data = final_df)
    st.subheader('Plot for predicted exchange rates')
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(1,1,figsize = (15,3))
    ax.plot(final_df.set_index('Business_Days'))
    plt.xticks(rotation='vertical')
    plt.ylabel("Exchange rate")
    #plt.show() 
    st.pyplot(fig)
    #st.line_chart(final_df.set_index('Business_Days'),ylim=(74,78))    
    st.write('Thanks','---', name )
        #st.line_chart(scaler.inverse_transform(lst_output))
    