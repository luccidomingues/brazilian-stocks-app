import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
import yfinance as yf

st.title('IBOVESPA App')

st.markdown("""
This app retrieves the list of the **IBOVESPA** (from toroinvestimentos) and its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** https://blog.toroinvestimentos.com.br/empresas-listadas-b3-bovespa.
""")

st.sidebar.header('User Input Features')





# Web scraping of IBOVESPA data
#

# @st.cache
# Luciano´s change
@st.cache(allow_output_mutation=True)


def load_data():
    url = 'https://blog.toroinvestimentos.com.br/empresas-listadas-b3-bovespa'
    html = pd.read_html(url, header = 0)
    df = html[1]
    return df

df = load_data()





# Luciano code
df['Setor'] = df['Setor'].str.replace('Bens Industriais','Bens industriais')
df['Setor'] = df['Setor'].str.replace('Energia Elétrica','Energia elétrica')
df.rename(columns={'Código da ação':'ticker'}, 
                 inplace=True)
# ------------------------

# sector = df.groupby('GICS Sector')
sector = df.groupby('Setor')



# Sidebar - Sector selection
sorted_sector_unique = sorted( df['Setor'].unique() )
selected_sector = st.sidebar.multiselect('Setor', sorted_sector_unique, sorted_sector_unique)



# Filtering data
# df_selected_sector = df[ (df['Setor'].isin(selected_sector)) ]
df_selected_sector = df[ (df.Setor.isin(selected_sector)) ]






st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')




# st.write('Lista ' + str(list(df_selected_sector[:10].Ticker+'.SA')))


st.dataframe(df_selected_sector)

# Download Ibovespa data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="IBOV.csv">Download CSV File</a>'
    return href



st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)



num_company = st.sidebar.slider('Number of Companies', 1, 10, 1, 1)








data = yf.download(
        tickers = "ABEV3.SA AZUL4.SA",
        period = "2y",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )





# https://pypi.org/project/yfinance/

# data = yf.download(
#         tickers = list(df_selected_sector[:10].Ticker+'.SA'),
    
# #     df.ticker + '.SA'
    
    
    
#         period = "2y",
#         interval = "1d",
#         group_by = 'Ticker',
#         auto_adjust = True,
#         prepost = True,
#         threads = True,
#         proxy = None
#     )


# data = yf.download(
#         tickers = list(df_selected_sector.Ticker+'.SA'),
#         period = "2y",
#         interval = "1d",
#         group_by = 'Ticker',
#         auto_adjust = True,
#         prepost = True,
#         threads = True,
#         proxy = None
#     )




# data = yf.download(  # or pdr.get_data_yahoo(...
#         # tickers list or string as well
#         tickers = "SPY AAPL MSFT",

#         # use "period" instead of start/end
#         # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#         # (optional, default is '1mo')
#         period = "ytd",

#         # fetch data by interval (including intraday if period < 60 days)
#         # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
#         # (optional, default is '1d')
#         interval = "1m",

#         # group by ticker (to access via data['SPY'])
#         # (optional, default is 'column')
#         group_by = 'ticker',

#         # adjust all OHLC automatically
#         # (optional, default is False)
#         auto_adjust = True,

#         # download pre/post regular market hours data
#         # (optional, default is False)
#         prepost = True,

#         # use threads for mass downloading? (True/False/Integer)
#         # (optional, default is True)
#         threads = True,

#         # proxy URL scheme use use when downloading?
#         # (optional, default is None)
#         proxy = None
#     )


# Luciano
st.set_option('deprecation.showPyplotGlobalUse', False)


# Plot Closing Price of Query Symbol
def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot()




# num_company = st.sidebar.slider('Number of Companies', 1, 10)
# num_company = st.sidebar.slider('Number of Companies', 1, 10, 1, 1)



if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i in list(df_selected_sector.Ticker+'.SA')[:num_company]:        
        price_plot(i)

        
        