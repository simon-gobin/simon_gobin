#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import coint
import scipy
from itertools import permutations


# In[2]:


def calculate_ols_trendline(y_values):
    x_values = np.arange(len(y_values))
    x_values = sm.add_constant(x_values)
    model = sm.OLS(y_values, x_values, missing='drop').fit()
    trendline = model.predict(x_values)
    return trendline


# In[3]:


def test_granger_causality(data_frame, column1, column2, max_lags=5):

    # Replace inf values with NaN and drop rows with NaN values
    data_frame = data_frame[[column1, column2]].replace([np.inf, -np.inf], np.nan).dropna()

    # Perform the Granger causality test
    test_result = grangercausalitytests(data_frame[[column1, column2]],max_lags, verbose=False)
    
    print("Granger Causality Test Results:\n")
    for lag in range(1, max_lags+1):
        f_test = test_result[lag][0]['ssr_ftest']
        chi2_test = test_result[lag][0]['ssr_chi2test']
        lr_test = test_result[lag][0]['lrtest']
        params_ftest = test_result[lag][0]['params_ftest']
        
        print(f"Lag {lag}:")
        print(f"  F-test: Statistic = {f_test[0]}, P-value = {f_test[1]}")
        print(f"  Chi-squared test: Statistic = {chi2_test[0]}, P-value = {chi2_test[1]}")
        print(f"  Likelihood-ratio test: Statistic = {lr_test[0]}, P-value = {lr_test[1]}")
        print(f"  Params F-test: Statistic = {params_ftest[0]}, P-value = {params_ftest[1]}\n")
    
    return test_result


# In[4]:


from itertools import permutations
import pandas as pd

def test_all_pairs_granger_causality(data_frame, max_lags=5):
    results = []
    columns = data_frame.columns
    for col1, col2 in permutations(columns, 2):
        data = data_frame[[col1, col2]].replace([np.inf, -np.inf], np.nan).dropna()
        test_result = grangercausalitytests(data, max_lags, verbose=False)
        
        for lag in range(1, max_lags+1):
            f_test = test_result[lag][0]['ssr_ftest']
            result = {
                'Causality Vallues': f'{col1}/{col2}',
                'Lag': lag,
                'F-Statistic': f_test[0],
                'P-value': f_test[1],
                'Significant': 'Yes' if f_test[1] < 0.05 else 'No'
            }
            results.append(result)
        
    granger_results_df = pd.DataFrame(results)
    return granger_results_df


# In[5]:


def t_test(data_frame, column1, column2, equal_var=False):
    
    # Replace inf values with NaN and drop rows with NaN values
    data_frame = data_frame[[column1, column2]].replace([np.inf, -np.inf], np.nan).dropna()
    
    f_statistic, p_value = scipy.stats.ttest_ind(data_frame[column1], data_frame[column2])
    
    if p_value < 0.05:
        print("There is a significant difference between the groups.")
    else:
        print("No significant difference found between the groups.")
        
    return f_statistic, p_value


# In[6]:


def test_levene(data_frame, column1, column2):

    # Replace inf values with NaN and drop rows with NaN values
    data_frame = data_frame[[column1, column2]].replace([np.inf, -np.inf], np.nan).dropna()

    # Perform the levene test
    f_statistic, p_value = scipy.stats.levene(data_frame[column1], data_frame[column2] , center='median', proportiontocut=0.05)
    
    print(f"Levene Test F-statistic: {f_statistic}")
    print(f"P-value: {p_value}")
    
    if p_value < 0.05:
        print("There is a significant difference between the groups.")
    else:
        print("No significant difference found between the groups.")

        
    return f_statistic, p_value


# In[7]:


def test_anova(data_frame, columns,use_var='unequal',welch_correction=True):
    
    data_frame = data_frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    groups = [data_frame[col] for col in columns]
    f_statistic, p_value = scipy.stats.f_oneway(*groups)
    
    print(f"ANOVA Test F-statistic: {f_statistic}")
    print(f"P-value: {p_value}")
    
    if p_value < 0.05:
        print("There is a significant difference between the groups.")
    else:
        print("No significant difference found between the groups.")


    return f_statistic, p_value


# In[8]:


def test_coint(data_frame, column1, column2):

    # Replace inf values with NaN and drop rows with NaN values
    data_frame = data_frame[[column1, column2]].replace([np.inf, -np.inf], np.nan).dropna()

    # Perform the Cointegration test
    score, p_value, crit_values =  coint(data_frame[column1], data_frame[column2])
    
    print(f"Cointegration Test Statistic: {score}")
    print(f"P-value: {p_value}")
    if p_value < 0.05:
        print("The series are likely cointegrated.")
    else:
        print("No evidence of cointegration.")
    
    return score, p_value, crit_values


# In[9]:


def test_all_pairs_coint(data_frame):
    results = []
    columns = data_frame.columns
    for col1, col2 in permutations(columns, 2):
        data = data_frame[[col1, col2]].replace([np.inf, -np.inf], np.nan).dropna()
        if col1 == col2:
            continue
        else: 
            score, p_value, crit_values = test_coint(data_frame, col1, col2)
            result = {
            'Score': f'{col1}/{col2}',
            'Test Statistic': score,
            'P-value': p_value,
            'Cointegrated': 'Yes' if p_value < 0.05 else 'No'
            }
            results.append(result)
        
    coint_matrix = pd.DataFrame(results)
    
    return coint_matrix
       


# Data set import and cleaning

# In[10]:


euro = pd.read_csv('euro 5 years.csv')
bitcoin = pd.read_csv('BTC-USD 5 years.csv')
etherium = pd.read_csv('ETH-USD 5 years.csv')
sp = pd.read_csv('S&P 500 (SPX) Historical Data NASDAQ.csv')
cac = pd.read_csv('cac 40 5 years.csv')
fts = pd.read_csv('FTSE 100 5 years.csv')
pound = pd.read_csv('GBP_USD 5 years.csv')


# In[11]:


fts[['Open', 'High', 'Low']] = fts[['Open', 'High', 'Low']].replace(',', '', regex=True).astype('float64')
cac[['Open', 'High', 'Low']] = cac[[' Open', ' High', ' Low']].replace(',', '', regex=True).astype('float64')


# Function : convert same format Date, calculate the mean, concact the mean vallue in a new Data Frame and
# replace the NaN with linar interpolation methode

# In[12]:


def process_dataframes(dataframes, date_formats):
    processed_dataframes = {}


    for name, df in dataframes.items():
       
        #Convert the Data set in same format and thest the Date in index
        df['Date'] = pd.to_datetime(df['Date'], format=date_formats.get(name), errors='coerce')
        df.set_index('Date', inplace=True)

        # Calculate mean prices
        mean_col_name = f'Mean Price {name.capitalize()}'
        df[mean_col_name] = df[['Open', 'High', 'Low']].mean(axis=1)
        processed_dataframes[name] = df[mean_col_name]

    # Concatenate mean prices in new DataFrame
    all_data_mean = pd.concat(processed_dataframes.values(), axis=1)
    all_data_mean.columns = processed_dataframes.keys()
    
    #Convert CAC 40 and FTSE 100 in dolars and replace the actual vallue
    all_data_mean['cac'] = all_data_mean['cac'] * all_data_mean['euro']
    all_data_mean['fts'] = all_data_mean['fts'] * all_data_mean['pound']
    
    #Remove the remplace vallue with interpolate function
    all_data_mean = all_data_mean[all_data_mean.columns].interpolate(method= 'linear' )



    return all_data_mean

dataframes = {'euro': euro, 'bitcoin': bitcoin, 'etherium': etherium, 'fts': fts, 'pound': pound, 
              'sp': sp, 'cac': cac}
date_formats = {'euro': None, 'bitcoint': None, 'etherium': None, 'fts': '%d/%m/%Y', 
                'pound': '%d/%m/%Y', 'sp': '%m/%d/%Y', 'cac': '%m/%d/%y'}


all_data_mean = process_dataframes(dataframes, date_formats)


# In[13]:


all_data_mean.columns


# In[14]:


all_data_mean.info()


# # Bitcoint/Etherium analyse 

# In[15]:


all_data_mean[['bitcoin', 'etherium']].corr(method= 'spearman')


# In[16]:


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), sharex=False)

# Plot the mean price of Bitcoin
all_data_mean['bitcoin'].plot(ax=ax1, fontsize=16, color='blue')
ax1.set_ylabel('Mean Price Bitcoin')
ax1.set_title('Mean Prices of Bitcoin and Ethereum with Rolling Correlation')

# Plot the mean price of Ethereum
all_data_mean['etherium'].plot(ax=ax2, fontsize=16, color='orange')
ax2.set_ylabel('Mean Price Ethereum')

# Calculate and plot the rolling correlation in the third subplot
rolling_corr = all_data_mean['etherium'].rolling(30).corr(all_data_mean['bitcoin'])
rolling_corr.plot(ax=ax3, fontsize=16, color='green')
ax3.set_ylabel('Rolling Correlation')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)

# Add legend to the third subplot
ax1.legend(['Bitcoin Mean Price'])
ax2.legend(['Bitcoin Mean Etherium'])
ax3.legend(['Rolling Correlation BTC/ETH'])

# Use the seaborn style
plt.style.use('seaborn-v0_8-colorblind')

plt.savefig('bitcoin_etheriun.jpeg')

# Show the plots
plt.show()


# Make the Granger Causality score beetwenn Bitcoin and Ehterium

# In[17]:


test_granger_causality(all_data_mean, 'bitcoin', 'etherium', max_lags=5)


# Caluclate the Covariance vallue for now if I can run a classic T-test or a Walsh t-test

# In[18]:


plt.figure(figsize=(8, 6))
sns.heatmap(all_data_mean[['bitcoin', 'etherium']].cov(), annot=True)
plt.title('Heatmap Covariance Bitcoin - Etherium')


# Calculate the cointegration score for Bitcoin and Etherium

# In[19]:


test_coint(all_data_mean, 'bitcoin', 'etherium')


# Because the covaraince are not equal i made a Welch t-test

# In[20]:


t_test(all_data_mean, 'bitcoin', 'etherium', equal_var=False)


# I compare the mean vallue of Bitcoin and Ehterium 

# In[21]:


all_data_mean['bitcoin'].mean()


# In[22]:


all_data_mean['etherium'].mean()


# # Analyse S&P/CAC/FTSE

# Make the Granger Causality score beetwenn S&P/CAC4/FTSE 100

# In[23]:


test_granger_causality(all_data_mean, 'sp', 'fts', max_lags=5)


# In[24]:


test_granger_causality(all_data_mean, 'sp', 'cac', max_lags=5)


# Create a heatmap with the correlation score and a line plot of the 3 vallue for see the evolution during the time.

# In[25]:


plt.figure(figsize=(8, 6))
sns.heatmap(all_data_mean[['cac', 'fts','sp']].corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap Correlation CAC 40/S&P/FTSE 100')


# In[26]:


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15))

# Plot for 'Mean Price CAC 40'
all_data_mean['cac'].plot(ax=ax1,fontsize=16, color='blue')
ax1.set_title('Mean Price CAC 40')
ax1.legend()

# Plot for 'Mean Price FTSE 100'
all_data_mean['fts'].plot(ax=ax2, fontsize=16, color='orange')
ax2.set_title('Mean Price FTSE 100')
ax2.legend()

# Plot for 'Mean Price S&P 500'
all_data_mean['sp'].plot(ax=ax3,fontsize=16 , color='green')
ax3.set_title('Mean Price S&P 500')
ax3.legend()


# I have create a Correlation line wiht statsmodels for visualise the mean evolution of the correaltion score during time.

# In[27]:


corr_ftse_cac = all_data_mean['fts'].rolling(30).corr(all_data_mean['cac'])
corr_ftse_sp = all_data_mean['fts'].rolling(30).corr(all_data_mean['sp'])
corr_cac_sp = all_data_mean['cac'].rolling(30).corr(all_data_mean['sp'])

fig = make_subplots(rows=3, cols=1)

corr_names = ["FTSE 100 vs CAC 40", "FTSE 100 vs S&P 500", "CAC 40 vs S&P 500"]


# Add traces and trendlines
for i, (corr, name) in enumerate(zip([corr_ftse_cac, corr_ftse_sp, corr_cac_sp], corr_names), 1):
    fig.add_trace(go.Scatter(x=all_data_mean.index, y=corr, mode='lines', name=name), row=i, col=1)
    fig.add_trace(go.Scatter(x=all_data_mean.index, y=calculate_ols_trendline(corr), mode='lines', name=f'{name} Trendline', line=dict(color='red')), row=i, col=1)

fig.update_layout(title='Rolling Correlations Over Time with OLS Trendlines', height=600, showlegend=True)
fig.update_xaxes(title_text='Date', row=3, col=1)
fig.update_yaxes(title_text='Correlation')

fig.show()


# Like for bitcoint I have create a covarianse heatmap for now if i can use the clasic Anova test or the Welch variation.

# In[28]:


plt.figure(figsize=(8, 6))
sns.heatmap(all_data_mean[['sp', 'fts','cac']].cov(), annot=True)
plt.title('Heatmap Covariance CAC 40/S&P/FTSE 100')


# In[29]:


test_anova(all_data_mean, ['fts', 'cac','sp'],use_var='unequal',welch_correction=True)


# To be sure of my Anova result I run three Welsh t-test to see if the result are the same 

# In[30]:


t_test(all_data_mean, 'sp', 'cac', equal_var=False)


# In[31]:


t_test(all_data_mean, 'fts', 'cac', equal_var=False)


# In[32]:


t_test(all_data_mean, 'fts', 'sp', equal_var=False)


# I use the Levene test to compare the Variance betwen the group and validate my Covariance tab result

# In[33]:


test_levene(all_data_mean, 'fts', 'cac')


# In[34]:


test_levene(all_data_mean, 'fts', 'sp')


# In[35]:


test_levene(all_data_mean, 'sp', 'cac')


# # Analyse all market together

# For this part I have make scatter matrix for compare the varaition of all data together and a heat map for compare the correlation score.

# In[36]:


sns.pairplot(all_data_mean,  kind='reg', diag_kind= 'kde', plot_kws=dict(marker="+", line_kws={'color': 'red'}))


# In[37]:


sns.pairplot(all_data_mean.pct_change(),  kind='reg', diag_kind= 'kde', plot_kws=dict(marker="+", line_kws={'color': 'red'}))


# In[38]:


plt.figure(figsize=(8, 6))
sns.heatmap(all_data_mean.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap Correlation')


# In[39]:


plt.figure(figsize=(8, 6))
sns.heatmap(all_data_mean.pct_change().corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap Correlation Percentage Variation')


# I make Cointegration test and Grange for compare all the data together and create a scatter plot matrix to visuliase the result

# In[40]:


coint_matrix = test_all_pairs_coint(all_data_mean)
coint_matrix.sort_values(by='P-value')


# In[41]:


granger = test_all_pairs_granger_causality(all_data_mean, max_lags=5)


# In[42]:


fig= px.scatter(coint_matrix, x='Score',y='P-value', title="Cointvariance result")
fig.add_hline(y=0.05, line_dash="dash", line_color="red")


# In[43]:


fig= px.scatter(granger, x='Causality Vallues',y='P-value', title="Granger causality result",color="Lag")
fig.add_hline(y=0.05, line_dash="dash", line_color="red")


# I found a big cointegration score that mean a bit link between the Ehterium and FTSE 100 so i create a Plot line fro visulise this result

# In[44]:


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), sharex=False)


all_data_mean['etherium'].plot(ax=ax1, fontsize=16, color='blue')
ax1.set_ylabel('Mean Price Etherium')
ax1.set_title('Mean Prices of Etherium and FTSE 100 with Rolling Correlation')


all_data_mean['fts'].plot(ax=ax2, fontsize=16, color='orange')
ax2.set_ylabel('Mean Price FTS 100')


rolling_corr = all_data_mean['fts'].rolling(30).corr(all_data_mean['bitcoin'])
rolling_corr.plot(ax=ax3, fontsize=16, color='green')
ax3.set_ylabel('Rolling Correlation')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)


ax1.legend(['Etherium Mean Price'])
ax2.legend(['FTSE 100 Mean Price'])
ax3.legend(['Rolling Correlation FTSE/BTC'])

plt.style.use('seaborn-v0_8-colorblind')


# # Majore variaiton 

# I have extract the 50 largest varaition in positive and 50 largest varaiton in negative and create a new datframe with this vallue

# In[45]:


top_10_largest = all_data_mean.pct_change().apply(lambda col: col.nlargest(50))
top_10_smallest = all_data_mean.pct_change().apply(lambda col: col.nsmallest(50))
top10 = top_10_largest.copy()
top10_2=top_10_smallest.copy()
df_top = pd.concat([top10, top10_2])


# I need to convert the new data frame in long fromat for create a historam with plotly

# In[46]:


# Convert the DataFrame to a long format
df_long = df_top.reset_index().melt(id_vars='Date', var_name='Variable', value_name='Value')

# Create a scatter plot with a separate facet for each variable
fig = px.histogram(df_long, x='Date', y='Value', facet_row='Variable', title="Majore variation of price",nbins=260)

# Update layout for better readability
fig.update_layout(height=1200, showlegend=False)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(matches=None) 

fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




