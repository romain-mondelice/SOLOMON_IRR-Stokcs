# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:29:17 2021
@author: Romain
"""

"""
############################################################################################################################################################
# Init of Solomon
# Cron job, every day when the US and EU market are open
# While loop :
# Get most recent data, convert it to gaf, do the prediction, and put a trade
############################################################################################################################################################
"""

import pandas as pd
import numpy as np

import pytz
import tensorflow as tf
import datetime
import datetime

from sklearn.preprocessing import MinMaxScaler

import MetaTrader5 as mt5
from talib import RSI

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
symbols = ["#AMZN", "#GOOG", "#AAPL", "#KER", "#RMS", "#MC.FR"]  #Validated stocks, #AMZN", "#GOOG", "#AAPL", "#KER", "#RMS", "#MC.FR"

stocks_actions = []

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
        
    date = datetime.datetime.now()
    
    #Here the year / month / day / hour / minutes that we are going to
    #See utc_from / utc_to
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minutes = date.minute
    
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
    
    df = MAIN_df.iloc[:-1].copy()
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
    
    train_df['RSI'] = RSI(train_df['Close'].values, timeperiod=14)
    current_rsi = train_df['RSI'].iloc[-1]
    
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
    train_df = train_df.drop(columns=['Date', 'RSI', 'Min', 'Max', 'Pct Change Min', 'Pct Change Max', 'Up Ratio', 'Down Ratio'])
    
    # =============================================================================
    # to predict df    
    # =============================================================================
    
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
    
    preds_list_total = []
    preds_list_total.append([preds_list_trimester[-1][0]])
    
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
    
    """
    # =============================================================================
    # Check the current position and take action in consequence
    # =============================================================================
    """
    
    #Get the current position
    # establish connection to the MetaTrader 5 terminal
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
     
    # get open positions on symbol
    positions=mt5.positions_get(symbol=symbol)
    
    #Buy type = 0
    #Sell type = 1
    print("=============================================================================")
    print(f"= {positions}")
    print("=============================================================================")
    
    #Send me an email for more information (mondelice.romain@gmail.com)
    
    stocks_actions.append([preds_, symbol])
    
"""
# =============================================================================
# The stocks to buy or to sell
# =============================================================================
"""

def open_position(symbol, lot, order_type, comment):
    price = mt5.symbol_info_tick(symbol).ask
    deviation = 20
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": deviation,
        "magic": 234000,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
    }
     
    # send a trading request
    result = mt5.order_send(request)
    print("=============================================================================")
    print("= order_send done, ", result)
    print("=============================================================================\n")

def close_position(symbol, lot, order_type, comment, position_id):
    price = mt5.symbol_info_tick(symbol).ask
    deviation = 20
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "position": position_id,
        "price": price,
        "deviation": deviation,
        "magic": 234000,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
    }
    
    # send a trading request
    result = mt5.order_send(request)
    print("=============================================================================")
    print("= order_send done, ", result)
    print("=============================================================================\n")

positions=mt5.positions_get()
current_stocks = []
current_stocks_lot = []
for position in positions:
    current_stocks.append(position.symbol)    
    current_stocks_lot.append(position.volume)
    
stocks_to_buy = []
stocks_to_sell = []
stocks_to_pass = []

for elem in stocks_actions:
    print("\n=================================================================================================")
    print(f"SOLOMON prediction of INFERENCE Risk ratio for the current candle of {elem[1]}: {elem[0]}")
    print("=================================================================================================")
    if elem[0][0] == 1 and elem[1] not in current_stocks:
        stocks_to_buy.append(elem[1])
    elif elem[0] == 0 in current_stocks:
        stocks_to_sell.append(elem[1])
    else:
        stocks_to_pass.append(elem[1])
        
print("\n=================================================================================================")
print(f"SOLOMON count of buy stocks : {len(stocks_to_buy)} ---- stocks : {stocks_to_buy}")
print(f"SOLOMON count of sell stocks : {len(stocks_to_sell)} ---- stocks : {stocks_to_sell}")
print(f"SOLOMON count of pass stocks : {len(stocks_to_pass)} ---- stocks : {stocks_to_pass}")
print("=================================================================================================\n")

account_info_dict = mt5.account_info()._asdict()
maximum_of_share = account_info_dict['balance'] * 4.5

if(len(stocks_to_buy) >= 1):
    total_div = len(stocks_to_buy) + len(positions)
    amount_of_share = maximum_of_share / total_div
    #Sell the surplus
    for i, stock in enumerate(current_stocks):
        current_lot = current_stocks_lot[i]
        symbol_info_dict = mt5.symbol_info(stock)._asdict()
        current_price = symbol_info_dict['ask']
        total_value = current_price * current_lot
        lot_surplus = total_value - amount_of_share
        lot_surplus = lot_surplus / current_price
        close_position(stock, round(lot_surplus, 1), mt5.ORDER_TYPE_SELL, "close : solomon script C1-1", positions[i].identifier)
    
    #Buy the new stocks
    for i, stock in enumerate(stocks_to_buy):
        symbol_info_dict = mt5.symbol_info(stock)._asdict()
        current_price = symbol_info_dict['ask']
        lot = amount_of_share / current_price
        if(lot > 350):
            print("=================================================================================================")
            print("! INVALID LOT - LOT IS TO HIGH !")
            print("=================================================================================================")
        open_position(stock, round(lot, 1), mt5.ORDER_TYPE_BUY, "open : solomon script C1-1")

if(len(stocks_to_sell) >= 1):
    for i, stock in enumerate(stocks_to_sell):
        positions_stock = mt5.positions_get(symbol = stock)
        for position in positions_stock:
            lot = position.volume
            close_position(stock, round(lot, 1), mt5.ORDER_TYPE_SELL, "close : solomon script C2-1", position.identifier)


















