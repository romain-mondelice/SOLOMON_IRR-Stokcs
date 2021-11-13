# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:29:17 2021
@author: Romain
"""

"""
Implementation of Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin & Vedant Misra paper on finance data
GROKKING: GENERALIZATION BEYOND OVERFITTING ON SMALL ALGORITHMIC DATASETS
"""

import pandas as pd
import numpy as np

import pytz
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import MetaTrader5 as mt5

"""
##############################################################################
# HyperParameters => TODO (find the best combinaison of hyperparameters)
##############################################################################
"""

#Use to set the data lenhgt
year_lenght = 20

lookback_window_size = 360

#TimeFrame that we use for our FOREX data
timeframe = mt5.TIMEFRAME_D1

#Week size associated to the timeframe ex: H4 => 5 days => 1 week => 30 H4
size_week = 5 #Avec 0.1

#Symbol => If other that EURUSD should change commission rate for more precision
symbols = ["#AMZN"]  #"#AMZN", "#GOOG", "#KER", "#RMS", "#AAPL" have to be the same name that in AdmiralMarket

dates = [[2019, 1, 5], [2019, 1, 12], [2019, 1, 19], [2019, 1, 26],  
        [2019, 2, 2], [2019, 2, 9], [2019, 2, 16], [2019, 2, 23], 
        [2019, 3, 2], [2019, 3, 9], [2019, 3, 16], [2019, 3, 23], [2019, 3, 30], 
        [2019, 4, 6], [2019, 4, 13], [2019, 4, 20], [2019, 4, 27], 
        [2019, 5, 4], [2019, 5, 11], [2019, 5, 18], [2019, 5, 25],
        [2019, 6, 1], [2019, 6, 8], [2019, 6, 15], [2019, 6, 22], [2019, 6, 29], 
        [2019, 7, 6], [2019, 7, 13], [2019, 7, 20], [2019, 7, 27], 
        [2019, 8, 3], [2019, 8, 10], [2019, 8, 17], [2019, 8, 24], [2019, 8, 31],
        [2019, 9, 7], [2019, 9, 14], [2019, 9, 21], [2019, 9, 28],
        [2019, 10, 5], [2019, 10, 12], [2019, 10, 19], [2019, 10, 26], 
        [2019, 11, 2], [2019, 11, 9], [2019, 11, 16], [2019, 11, 23], [2019, 11, 30],
        [2019, 12, 7], [2019, 12, 14], [2019, 12, 21], [2019, 12, 28], 
        [2020, 1, 4], [2020, 1, 11], [2020, 1, 18], [2020, 1, 25],  
        [2020, 2, 1], [2020, 2, 8], [2020, 2, 15], [2020, 2, 22], [2020, 2, 29], 
        [2020, 3, 7], [2020, 3, 14], [2020, 3, 21], [2020, 3, 28], 
        [2020, 4, 4], [2020, 4, 11], [2020, 4, 18], [2020, 4, 25], 
        [2020, 5, 2], [2020, 5, 9], [2020, 5, 16], [2020, 5, 23], [2020, 5, 30],
        [2020, 6, 6], [2020, 6, 13], [2020, 6, 20], [2020, 6, 27], 
        [2020, 7, 4], [2020, 7, 11], [2020, 7, 18], [2020, 7, 25], 
        [2020, 8, 1], [2020, 8, 8], [2020, 8, 15], [2020, 8, 22], [2020, 8, 29],
        [2020, 9, 5], [2020, 9, 12], [2020, 9, 19], [2020, 9, 26],
        [2020, 10, 3], [2020, 10, 10], [2020, 10, 17], [2020, 10, 24], [2020, 10, 31],
        [2020, 11, 7], [2020, 11, 14], [2020, 11, 21], [2020, 11, 28],
        [2020, 12, 5], [2020, 12, 12], [2020, 12, 19], [2020, 12, 26],
        [2021, 1, 2], [2021, 1, 9], [2021, 1, 16], [2021, 1, 23], [2021, 1, 30],  
        [2021, 2, 6], [2021, 2, 13], [2021, 2, 20], [2021, 2, 27],
        [2021, 3, 6], [2021, 3, 13], [2021, 3, 20], [2021, 3, 27],
        [2021, 4, 3], [2021, 4, 10], [2021, 4, 17], [2021, 4, 24], 
        [2021, 5, 1], [2021, 5, 8], [2021, 5, 15], [2021, 5, 22],
        [2021, 5, 29], [2021, 6, 5], [2021, 6, 12], [2021, 6, 19], [2021, 6, 26], 
        [2021, 7, 3], [2021, 7, 10], [2021, 7, 17], [2021, 7, 24], [2021, 7, 31],
        [2021, 8, 7], [2021, 8, 14], [2021, 8, 21], [2021, 8, 28],
        [2021, 9, 4], [2021, 9, 11], [2021, 9, 18], [2021, 9, 25],
        [2021, 10, 2], [2021, 10, 9], [2021, 10, 16], [2021, 10, 23]]

"""
[2021, 1, 2], [2021, 1, 9], [2021, 1, 16], [2021, 1, 23], [2021, 1, 30],  
        [2021, 2, 6], [2021, 2, 13], [2021, 2, 20], [2021, 2, 27],
        [2021, 3, 6], [2021, 3, 13], [2021, 3, 20], [2021, 3, 27],
        [2021, 4, 3], [2021, 4, 10], [2021, 4, 17], [2021, 4, 24], 
        [2021, 5, 1], [2021, 5, 8], [2021, 5, 15], [2021, 5, 22],
        [2021, 5, 29], [2021, 6, 5], [2021, 6, 12], [2021, 6, 19], [2021, 6, 26], 
        [2021, 7, 3], [2021, 7, 10], [2021, 7, 17], [2021, 7, 24], [2021, 7, 31],
        [2021, 8, 7], [2021, 8, 14], [2021, 8, 21], [2021, 8, 28],
        [2021, 9, 4], [2021, 9, 11], [2021, 9, 18], [2021, 9, 25],
        [2021, 10, 2], [2021, 10, 9], [2021, 10, 16], [2021, 10, 23]
        
[2020, 1, 4], [2020, 1, 11], [2020, 1, 18], [2020, 1, 25],  
        [2020, 2, 1], [2020, 2, 8], [2020, 2, 15], [2020, 2, 22], [2020, 2, 29], 
        [2020, 3, 7], [2020, 3, 14], [2020, 3, 21], [2020, 3, 28], 
        [2020, 4, 4], [2020, 4, 11], [2020, 4, 18], [2020, 4, 25], 
        [2020, 5, 2], [2020, 5, 9], [2020, 5, 16], [2020, 5, 23], [2020, 5, 30],
        [2020, 6, 6], [2020, 6, 13], [2020, 6, 20], [2020, 6, 27], 
        [2020, 7, 4], [2020, 7, 11], [2020, 7, 18], [2020, 7, 25], 
        [2020, 8, 1], [2020, 8, 8], [2020, 8, 15], [2020, 8, 22], [2020, 8, 29],
        [2020, 9, 5], [2020, 9, 12], [2020, 9, 19], [2020, 9, 26],
        [2020, 10, 3], [2020, 10, 10], [2020, 10, 17], [2020, 10, 24], [2020, 10, 31],
        [2020, 11, 7], [2020, 11, 14], [2020, 11, 21], [2020, 11, 28],
        [2020, 12, 5], [2020, 12, 12], [2020, 12, 19], [2020, 12, 26]
        
[2019, 1, 5], [2019, 1, 12], [2019, 1, 19], [2019, 1, 26],  
        [2019, 2, 2], [2019, 2, 9], [2019, 2, 16], [2019, 2, 23], 
        [2019, 3, 2], [2019, 3, 9], [2019, 3, 16], [2019, 3, 23], [2019, 3, 30], 
        [2019, 4, 6], [2019, 4, 13], [2019, 4, 20], [2019, 4, 27], 
        [2019, 5, 4], [2019, 5, 11], [2019, 5, 18], [2019, 5, 25],
        [2019, 6, 1], [2019, 6, 8], [2019, 6, 15], [2019, 6, 22], [2019, 6, 29], 
        [2019, 7, 6], [2019, 7, 13], [2019, 7, 20], [2019, 7, 27], 
        [2019, 8, 3], [2019, 8, 10], [2019, 8, 17], [2019, 8, 24], [2019, 8, 31],
        [2019, 9, 7], [2019, 9, 14], [2019, 9, 21], [2019, 9, 28],
        [2019, 10, 5], [2019, 10, 12], [2019, 10, 19], [2019, 10, 26], 
        [2019, 11, 2], [2019, 11, 9], [2019, 11, 16], [2019, 11, 23], [2019, 11, 30],
        [2019, 12, 7], [2019, 12, 14], [2019, 12, 21], [2019, 12, 28]
"""

for symbol in symbols:
    all_opens = []
    all_low = []
    all_high = []
    all_closes = []
    
    preds_list_week = []
    preds_list_month = []
    preds_list_trimester = []
    preds_list_semester = []
    preds_list_year = []
    preds_list_total = []
    for date in dates:
        
        """
        ##############################################################################
        # Init of Solomon
        ##############################################################################
        """
        all_preds = []
        
        #Here the year / month / day / hour / minutes that we are going to
        #See utc_from / utc_to
        year = date[0]
        month = date[1]
        day = date[2]
        hour = 0
        minutes = 0
        
        print("\n=============================================================================")
        print(f"Inference Solomon - {day}-{month}-{year}")
        print("=============================================================================")
        
        # establish connection to MetaTrader 5 terminal
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            quit()
        
        # set time zone to UTC
        timezone = pytz.timezone("Etc/UTC")
        
        # create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
        utc_from = datetime.datetime(year-year_lenght, month, day, tzinfo=timezone)
        utc_to = datetime.datetime(year, month, day, hour, minutes, tzinfo=timezone)
        
        # EURUSD Dataframe
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
        
        rates_frame = pd.DataFrame(rates)
        
        rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
        
        MAIN_df = rates_frame
        
        MAIN_df = MAIN_df.drop(columns=['spread', 'real_volume'])
        
        """
        # =============================================================================
        # Define the data on wich we have trained in order to fit_transform on them
        # =============================================================================
        """
        
        df = MAIN_df.copy()
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date'] = df['Date'].values.astype(np.datetime64)
        df['Volume'] = df['Volume'].values.astype(int)
        
        # =============================================================================
        # Calculate the mean TP available
        # =============================================================================
        mean_high = (df["High"] - df["Close"]).mean()
        mean_low = (df["Close"] - df["Low"]).mean()
        
        """
        # =============================================================================
        # Define trimester train_df
        # =============================================================================
        """
        train_df = df.copy()
        
        # =============================================================================
        # Classify with next close    
        # =============================================================================
        
        #Get the min and max of the week following the current close price
        train_df["Min"] = train_df['Close'].shift(-(size_week*4)*3).rolling(window=(size_week*4)*3).min()
        train_df["Max"] = train_df['Close'].shift(-(size_week*4)*3).rolling(window=(size_week*4)*3).max()
        
        #Achieve a percentage change in price
        #Example for the max: If the percentage change is high it means that the high point of 
        # the following week is far from the current price close
        train_df["Pct Change Min"] = ((train_df["Min"] - train_df['Close']) / train_df['Close']) * 100
        train_df["Pct Change Max"] = ((train_df["Max"] - train_df['Close']) / train_df['Close']) * 100
        
        #Then we convert our price changes into percentages to get a better overview
        train_df["Up Ratio"] = (abs(train_df["Pct Change Max"]) / (abs(train_df["Pct Change Min"]) + abs(train_df["Pct Change Max"]))) * 100
        train_df["Down Ratio"] = (abs(train_df["Pct Change Min"]) / (abs(train_df["Pct Change Min"]) + abs(train_df["Pct Change Max"]))) * 100
        
        #Finally we choose a ratio that we will classify
        #Attention according to the chosen ratio (Up or Down) the actions are reversed
        train_df['INFERENCE-RiskRatio'] = train_df["Up Ratio"]
        
        #We drop the useless columns so as not to disturb the learning process with data that already come from the future
        train_df = train_df.drop(columns=['Date', 'Min', 'Max', 'Pct Change Min', 'Pct Change Max', 'Up Ratio', 'Down Ratio'])
        
        all_opens.append(train_df["Open"].iloc[-1:].values)
        all_closes.append(train_df["Close"].iloc[-1:].values)
        all_low.append(train_df["Low"].iloc[-1:].values)
        all_high.append(train_df["High"].iloc[-1:].values)
        
        # =============================================================================
        # to predict df    
        # =============================================================================
        
        to_predict_df_trimester = train_df.iloc[len(train_df)-1-lookback_window_size:]
        
        # =============================================================================
        # Drop nans    
        # =============================================================================
        
        train_df = train_df.dropna()
        
        # =============================================================================
        # Scale train_df    
        # =============================================================================
        
        price_sc = MinMaxScaler(feature_range=(0, 100))
        volume_sc = MinMaxScaler(feature_range=(0, 100))
        ir_sc = MinMaxScaler(feature_range=(0, 100))
        
        train_df['Open'] = price_sc.fit_transform(train_df["Open"].values.reshape(-1,1))
        train_df['High'] = price_sc.transform(train_df["High"].values.reshape(-1,1))
        train_df['Low'] = price_sc.transform(train_df["Low"].values.reshape(-1,1))
        train_df['Close'] = price_sc.transform(train_df["Close"].values.reshape(-1,1))
        train_df['Volume'] = volume_sc.fit_transform(train_df["Volume"].values.reshape(-1,1))
        
        train_df = train_df.dropna()
        train_df = train_df.reset_index(drop=True)
        
        # =============================================================================
        # Detrend df    
        # =============================================================================
        
        train_df['Open'] = train_df['Open'].diff()
        train_df['High'] = train_df['High'].diff()
        train_df['Low'] = train_df['Low'].diff()
        train_df['Close'] = train_df['Close'].diff()
        train_df['Volume'] = train_df['Volume'].diff()
        
        # =============================================================================
        # Scale to_predict_df    
        # =============================================================================
        
        to_predict_df_trimester = to_predict_df_trimester.drop(columns=['INFERENCE-RiskRatio'])
        
        to_predict_df_trimester['Open'] = price_sc.transform(to_predict_df_trimester["Open"].values.reshape(-1,1))
        to_predict_df_trimester['High'] = price_sc.transform(to_predict_df_trimester["High"].values.reshape(-1,1))
        to_predict_df_trimester['Low'] = price_sc.transform(to_predict_df_trimester["Low"].values.reshape(-1,1))
        to_predict_df_trimester['Close'] = price_sc.transform(to_predict_df_trimester["Close"].values.reshape(-1,1))
        to_predict_df_trimester['Volume'] = volume_sc.transform(to_predict_df_trimester["Volume"].values.reshape(-1,1))
        
        # =============================================================================
        # Detrend df    
        # =============================================================================
        
        to_predict_df_trimester['Open'] = to_predict_df_trimester['Open'].diff()
        to_predict_df_trimester['High'] = to_predict_df_trimester['High'].diff()
        to_predict_df_trimester['Low'] = to_predict_df_trimester['Low'].diff()
        to_predict_df_trimester['Close'] = to_predict_df_trimester['Close'].diff()
        to_predict_df_trimester['Volume'] = to_predict_df_trimester['Volume'].diff()
        
        # =============================================================================
        # Dropna
        # =============================================================================
        train_df['INFERENCE-RiskRatio'] = train_df['INFERENCE-RiskRatio'].astype(int)
        
        # =============================================================================
        # Reorder column
        # =============================================================================
        column_names = ["Open", "High", "Low", "Close", "Volume", "INFERENCE-RiskRatio"]
        train_df = train_df.reindex(columns=column_names)
        
        # =============================================================================
        # Save train_df 
        # =============================================================================
        train_df.to_csv('C:/Users/romai/Desktop/INFERENCE/PRODUITS/INFERENCE - SOLOMON/solomon/train_df.csv', index = False)
        
        # =============================================================================
        # Rename df 
        # =============================================================================
        train_df_trimester = train_df.copy()
        
        """
        ######################################################################################
        # build all the train_x and train_y
        ######################################################################################
        """
        
        # =============================================================================
        # Finish the preparation of train data TRIMESTER
        # =============================================================================
        train_array = np.array(train_df_trimester)
        final_train_array = train_array
        
        train_total_x = []
        train_total_y = []
        
        for i in range(lookback_window_size, len(final_train_array)):
            train_total_x.append(final_train_array[(i-(lookback_window_size))+1:i+1, :train_df.shape[1]-1])
            train_total_y.append(final_train_array[i, -1])
        
        total_train = list(zip(train_total_x, train_total_y))
    
        train_x, train_y = zip(*total_train)
        
        train_x_trimester = np.array(train_x)
        train_y_trimester = np.array(train_y)
        
        train_array = np.array(to_predict_df_trimester)
        final_train_array = train_array
        
        train_total_x = []
        
        for i in range(lookback_window_size, len(final_train_array)):
            train_total_x.append(final_train_array[(i-(lookback_window_size))+1:i+1, :])
        
        to_predict_x_trimester = np.array(train_total_x)
        
        """
        ######################################################################################
        # Build model TRIMESTER
        ######################################################################################
        """
        
        #Send me an email for more information (mondelice.romain@gmail.com)
        
        """
        ######################################################################################
        # Train model TRIMESTER
        ######################################################################################
        """
        
        #Send me an email for more information (mondelice.romain@gmail.com)
        
        """
        ######################################################################################
        # Concatenate predictions
        ######################################################################################
        """
        
        preds_list_total.append([preds_list_trimester[-1][0]])
        
    # =============================================================================
    # Create df from data to test strategy and get metrics
    # =============================================================================
    
    irr_trimester = []
    
    for preds in preds_list_total:
        irr_trimester.append(preds[0])
    
    all_opens_merged = [item for sublist in all_opens for item in sublist]
    all_closes_merged = [item for sublist in all_closes for item in sublist]
    all_high_merged = [item for sublist in all_high for item in sublist]
    all_low_merged = [item for sublist in all_low for item in sublist]
    
    df_closes = pd.DataFrame(all_opens_merged, columns=['Open'])
    df_closes["High"] = all_high_merged
    df_closes["Low"] = all_low_merged
    df_closes["Close"] = all_closes_merged
    df_closes['IRR-TRIMESTER'] = irr_trimester
    df_closes['IRR-TRIMESTER'] = df_closes['IRR-TRIMESTER'].astype(float)
    
    # =============================================================================
    # Save df_closes 
    # =============================================================================
    
    #df_closes.to_excel(f'C:/Users/romai/Desktop/INFERENCE/PRODUITS/INFERENCE - SOLOMON/solomon/{symbol}-preds_df.xlsx')
    
    total = 0
    symbols = ["#AMZN", "#GOOG", "#AAPL", "#KER", "#RMS", "#MC.FR"]  #Validated stocks, #AMZN", "#GOOG", "#AAPL", "#KER", "#RMS", "#MC.FR"
    for symbol in symbols:
        #symbol = '#AMZN'
        df_closes = pd.read_excel(f'C:/Users/romai/Desktop/INFERENCE/PRODUITS/INFERENCE - SOLOMON/solomon/{symbol}-preds_df.xlsx')
        df_closes = df_closes.drop(columns=['Unnamed: 0'])
        
        df_closes["TRIMESTER_Mean"] = df_closes["IRR-TRIMESTER"].rolling(5).mean()
        
        from talib import RSI
        df_closes['RSI'] = RSI(df_closes['Close'].values, timeperiod=14)
        
        # =============================================================================
        # Classify with next close    
        # =============================================================================
        
        #Get the min and max of the week following the current close price
        df_closes["Min"] = df_closes['Close'].shift(-(size_week*4)*3).rolling(window=(size_week*4)*3).min()
        df_closes["Max"] = df_closes['Close'].shift(-(size_week*4)*3).rolling(window=(size_week*4)*3).max()
        
        #Achieve a percentage change in price
        #Example for the max: If the percentage change is high it means that the high point of 
        # the following week is far from the current price close
        df_closes["Pct Change Min"] = ((df_closes["Min"] - df_closes['Close']) / df_closes['Close']) * 100
        df_closes["Pct Change Max"] = ((df_closes["Max"] - df_closes['Close']) / df_closes['Close']) * 100
        
        #Then we convert our price changes into percentages to get a better overview
        df_closes["Up Ratio"] = (abs(df_closes["Pct Change Max"]) / (abs(df_closes["Pct Change Min"]) + abs(df_closes["Pct Change Max"]))) * 100
        df_closes["Down Ratio"] = (abs(df_closes["Pct Change Min"]) / (abs(df_closes["Pct Change Min"]) + abs(df_closes["Pct Change Max"]))) * 100
        
        #Finally we choose a ratio that we will classify
        #Attention according to the chosen ratio (Up or Down) the actions are reversed
        df_closes['INFERENCE-RiskRatio'] = df_closes["Up Ratio"]
        
        #We drop the useless columns so as not to disturb the learning process with data that already come from the future
        df_closes = df_closes.drop(columns=['Min', 'Max', 'Pct Change Min', 'Pct Change Max', 'Up Ratio', 'Down Ratio'])
        
        # =============================================================================
        # Calculate cumulative returns
        # =============================================================================
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.plot(df_closes['Close'], "-p", label="current_close")
        
        current_pos = None
        short_close = None
        long_close = None
        count_days = 0
        cumulutative_return = []
        
        """
        ######################################################################################
        # Use the strategy
        ######################################################################################
        """
        #Send me an email for more information (mondelice.romain@gmail.com)
        
        import statistics
        volatility = int(statistics.stdev(df_closes['Close'].values) * 100)
        total_cumulutative_return = int(sum(cumulutative_return) * 100)
        sharpe = total_cumulutative_return / volatility
        
        print("--------------------------------------------------------------------------")
        print(f"- Cumulative Returns on {symbol} : {total_cumulutative_return} points")
        print(f"- Mean Cumulative Returns by week : {int(total_cumulutative_return/len(df_closes))} points")
        print(f"- Sharpe Ratio : {round(sharpe, 5)}")
        print("--------------------------------------------------------------------------\n")
        
        plt.ylabel('Prices')
        plt.xlabel('Timesteps')
        plt.title(f'{symbol} Final test graph {round(sharpe, 5)} - {round(total_cumulutative_return, 5)}')
        plt.savefig(f"C:/Users/romai/Desktop/INFERENCE/PRODUITS/INFERENCE - SOLOMON/solomon/Plot reward mean over episode/{symbol}-final-test_plot_2019-2021.png")
        
        total += total_cumulutative_return
        print(total)