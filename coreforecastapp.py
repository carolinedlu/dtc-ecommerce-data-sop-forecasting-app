import streamlit as st
import base64
import pandas as pd
import matplotlib.pyplot as plt
import timeSeriesObject, analyse, transform, inference, metrics
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from pyxlsb import open_workbook as open_xlsb
st.set_option('deprecation.showPyplotGlobalUse', False) # haha, lol!

st.title('Automated Time Series Forecasting')

# Data Input
# Endogenous Variable Input (Select by Filters)
#try:
uploaded_file = st.file_uploader(label = 'Choose the database file')
data = pd.read_csv(uploaded_file)
st.header('This is how the complete dataset looks like')
st.write(data)
# Filter columns
filter_column_1 = st.selectbox('Select the product column to filter on' ,data.columns.tolist())
filter_column_1_value = st.selectbox('Select the product name you want',data[filter_column_1].unique())


data = data[(data[filter_column_1] == filter_column_1_value)].drop([filter_column_1], axis = 1)

st.header('This is how the sliced dataset looks like')
st.write(data)

# Problem Definition
st.header('Specify the time-series problem')
date_column = st.selectbox('What is the Date Column', (data.columns.tolist()))
date_format = st.selectbox('What is the Date Format?', ['%Y-%m-%d', '%d-%m-%Y','%Y/%m/%d','%Y%m%d','%d/%m/%Y','%d%m%Y',
                                                        '%Y-%d-%m', '%m-%d-%Y','%Y/%d/%m','%Y%d%m','%m/%d/%Y','%m%d%Y','%Y%m'])
try:
    data[date_column] = pd.to_datetime(data[date_column], format = date_format)
    st.markdown('Date correctly parsed! ')
except:
    st.markdown('Please re-check your date format and enter! \nThe program cannot proceed otherwise')

response_variable = st.selectbox('What is the Response Variable', (list(set(data.columns.tolist()) - set(date_column))))

#get best model file
#get best model
#do feature engineering
#generate predictions

start_train_date = data[date_column].min()
st.sidebar.markdown('Historic Data Start Date: \n\n{}'.format(start_train_date.strftime(format = '%Y-%m-%d')))

st.sidebar.header('Specify prediction time interval')
start_pred_date = st.sidebar.date_input(label = 'Prediction Start',value=data[date_column].min(), min_value= data[date_column].min(), max_value=data[date_column].max(), key = 5)
st.sidebar.markdown('This is the date from which your model will generate unseen predictions. \n\n')


end_pred_date = data[date_column].max()
st.sidebar.markdown('Prediction End Date: \n\n{}'.format(end_pred_date.strftime(format = '%Y-%m-%d')))
st.sidebar.markdown('This is the date till which your model will generate unseen predictions.')


## Forecasting Framework
# Pass the data
ts = timeSeriesObject.TimeSeriesObject(data)
# Set the date column
ts.set_date_column(date_column = date_column)
# Add or remove columns

ts.response_lag(date_column = date_column,response_variable = response_variable)
ts.response_rolling(date_column = date_column,response_variable = response_variable)
fft_modify = False
#st.markdown('Tick this box if you want to input peak sales patterns based on past data')
ts.fft_features(date_column = date_column,response_variable = response_variable,modify = fft_modify)
ts.date_features(date_column = date_column)
# Set the response variable
ts.xy_split(response_variable = response_variable)
# Display the time-series
st.header('Plot of the response variable with time')
plt.figure(figsize = (14,6), dpi = 80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size':14})
plt.plot(ts.y)
plt.xlabel('Dates')
plt.ylabel(response_variable)
st.pyplot()

# Transformations
#st.header('Specify automated transformations to use')
ts_transform = transform.Transform(ts)
boxcox_modify = False
#st.markdown('Tick this box if the variance of your time series is changing with time')
ts_transform.do_boxcox(modify = boxcox_modify)

stationarize_modify = False
#st.markdown('Tick this box if the time series has a visible trend')
ts_transform.stationarize_y(modify = stationarize_modify)
stationarize_x_modify = False
#st.markdown('You may try checking as well as unchecking this box to see which gives better results')
if stationarize_x_modify:
    ts_transform.stationarize_X()
scale_modify_y = False
#st.markdown('Tick this box if your response needs to be scaled')
scale_modify_x = False
#st.markdown('Tick this box if your data needs to be scaled')
scaling_choice = 'standard'
ts_transform.scaling_X(modify = scale_modify_x,scaling_method=scaling_choice)
ts_transform.scaling_y(modify = scale_modify_y,scaling_method=scaling_choice)
#st.header('Specify evaluation metric')
criteria = 'smape'
#st.markdown('Select "mape" if your time series has no zero values')
#st.markdown('Select "smape" if your time series has many zero values')
# Train-test-prediction split
ts.train_test_split(start_train_date = start_train_date,
                    end_train_date = start_pred_date - timedelta(days=17),
                    start_test_date = start_pred_date - timedelta(days=16),
                    end_test_date = start_pred_date - timedelta(days=1),
                    start_pred_date = start_pred_date,
                    end_pred_date = end_pred_date)
# Modelling

models = inference.Model(tso = ts, tst = ts_transform)

summary_table = models.load_summary_table(filter_column_1_value)
# Output Summary Table
st.header('Model Performance & Accuracy Metrics')
st.write(summary_table.drop('predictions', axis = 1))

# Best Model
best_model_name = models.find_best_model(summary_table, criteria = 'smape',)
best_results = models.get_inference_results(best_model_name, filter_column_1_value)
# Best Model Predictions Plot
st.header('Best Model Forecast and Actuals Plot')
percent_adjust = st.slider(label = 'Forecast Uplift Percentage', value=0.0, min_value=-200.0, max_value = 200.0)
# Preparing best predictions dataframe
print(best_results.predictions.values[0])
best_predictions = pd.DataFrame(data = best_results.predictions.values[0],
                                index = pd.date_range(start_pred_date, end_pred_date, freq = 'W-MON')).rename(columns={0:'kegs_sales'})
best_predictions.index = pd.to_datetime(best_predictions.index, format = '%Y-%m-%d')
best_predictions = best_predictions*(1+0.01*percent_adjust)
# Preparing historical dataframe
h = data[[date_column,response_variable]].set_index(date_column)
h.index = pd.to_datetime(h.index, format = '%d-%m-%Y')
# Plotting
plt.figure(figsize = (14,6), dpi = 80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size':14})
plt.plot(h, label = 'historicals')
plt.plot(best_predictions, label = 'predictions')
plt.xlabel('Dates')
plt.ylabel(response_variable)
plt.title('Forecast of {} as per the best model'.format(response_variable))
plt.legend()
st.pyplot()

# Predictions Download
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data
df_xlsx = to_excel(best_results)
st.download_button(label='Download Result',
                                data=df_xlsx ,
                                file_name= 'df_test.xlsx')

# Footer
st.markdown('***')
st.markdown('***')
st.markdown('***')
st.markdown('***')
st.markdown('For additional feature requests, feedback or further information regarding the automated time series forecasting tool, feel free to contact the ZX DS team')
#except:
#    st.markdown('Loop')
#    pass

