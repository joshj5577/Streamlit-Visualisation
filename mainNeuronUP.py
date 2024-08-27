import streamlit as st
st.title("Patient activity analysis")

from asyncio.log import logger
import base64
import binascii
from inspect import _empty
import requests
import streamlit as st
import pandas as pd
import numpy as np
pip install seaborn
import seaborn as sns
from matplotlib import pyplot as plt

neuronUP_5colors = ['#66e9ff',
                   '#00cdef','#00abc8','#0089a1','#00798d']
neuronUP_10colors_darktolight = ['#00798d','#0089a1','#009ab4','#00abc8','#00bcdc','#00cdef','#04dbff','#66e9ff','#7aecff','#8deeff']
neuronUP_12colors = ['#00798d','#0089a1','#009ab4','#00abc8','#00bcdc','#00cdef','#04dbff','#66e9ff','#7aecff','#8deeff', '#b4f4ff', '#dcfaff']
neuronup_pie_colors = ['#00798d','#00abc8', '#04dbff', '#8deeff']
neuronup_2pie_colors = ['#00abc8', '#0089a1']
neuronUP_10colors = ['#8deeff','#7aecff','#66e9ff','#04dbff',
                   '#00cdef','#00bcdc','#00abc8','#009ab4','#0089a1','#00798d']

def import_data(year):
  data_x = pd.read_csv('C:/Users/joshj/AA_NEURON UP INTERNSHIP/data_%d.csv' % year)
  return data_x

def import_activity(x):
  activity_x = pd.read_csv('C:/Users/joshj/INTERNSHIP DATA/All data/activity_%d.csv' % x)
  activity_x['activity_id'] = np.repeat(x, len(activity_x))
  return activity_x

#"""##  activity type function"""

def activity_types(x):
  activity_x = import_activity(x)
  if activity_x ['type'].iloc[0] == 'game':
   
    return st.write('Game type')
  elif activity_x ['type'].iloc[0] == 'card':
    
    
    return st.write('Card type')
  else:
    
    
    return st.write('Generator type')

def activity_type(x):
  activity_x = import_activity(x)
  if activity_x ['type'].iloc[0] == 'game':
   return ('Game type')
  elif activity_x ['type'].iloc[0] == 'card':
    return ('Card type')
  else:
    return ('Generator type')    

#"""## times played Histogram function"""

def times_played(x):
  activity_x = import_activity(x)
  st.write('total times played: ', len(activity_x))
  patient_times_played = activity_x['patient_id'].value_counts()
  times_played = pd.DataFrame()
  times_played['times_played'] = patient_times_played
  times_played = pd.DataFrame()
  times_played['times_played'] = patient_times_played
  times_played_over60 = times_played[times_played['times_played'] > 60]
  times_played_over60_below2500 = times_played_over60[times_played_over60['times_played']<2500]
  times_played = pd.DataFrame()
  times_played['times_played'] = patient_times_played
  times_played_upto60 = times_played[times_played['times_played'] <= 60]
  fig, axes = plt.subplots(1, 2, figsize=(25, 12))
  fig.suptitle('amount of users who played activity below v over 60 times ', fontsize=25)
  label1 = 'n (users)= ' + str(len(times_played_upto60)) + '\n '
  label2 = 'n (users) = ' + str(len(times_played_over60_below2500)) + '\n '
  ax = sns.histplot(ax=axes[0], data=times_played_upto60['times_played'], label=label1, color='#00ABC8')
  axes[0].set_title('Distribution of users playing activity %d <=60 times' %x, fontsize=25, y=1)
  ax.set_xlabel('Times played', fontsize=20)
  ax.set_ylabel('Count', fontsize=20)
  ax.xaxis.set_tick_params(labelsize='large')
  ax.yaxis.set_tick_params(labelsize='large')
  ax.legend(loc='best', fontsize=20)
  ax = sns.histplot(ax=axes[1], data=times_played_over60_below2500['times_played'],label=label2, color='#00ABC8')
  if times_played_over60['times_played'].describe().loc['max'] > 2500:
    axes[1].set_title('Distribution of users playing activity %d over 60 times (outliers over 2500 removed)'%x, y=1, fontsize=25)
  else:
    axes[1].set_title('Distribution of users playing activity %d over 60 times'%x, y=1, fontsize=25)
  ax.set_xlabel('Times played', fontsize=20)
  ax.set_ylabel('Count', fontsize=20)
  ax.xaxis.set_tick_params(labelsize='large')
  ax.yaxis.set_tick_params(labelsize='large')
  ax.legend(loc='best', fontsize=20)
  plt.subplots_adjust(wspace=0.3)
  unique, counts = np.unique(times_played['times_played'], return_counts=True)
  result = np.column_stack((unique, counts)) 
  df_count = pd.DataFrame(result, columns=['count', 'times_played'])
  st.pyplot(fig)
  return df_count.describe()

#"""
## times played by year bar chart function"""

def times_played_year (x):
  activity_x = import_activity(x)
  activity_x['date'] = pd.to_datetime(activity_x['date']).copy()
  activity_x['year']= activity_x['date'].dt.year.copy()
             
  by_yearx = activity_x.groupby('year')
  year_countx = by_yearx.aggregate(np.count_nonzero)
  year_countx = year_countx[['patient_id']]
  year_countx= year_countx.rename(columns={'patient_id':'Activities_played'}) 
  st.write(year_countx.sort_values(by='Activities_played', ascending=False))

  year_countx = year_countx.reset_index()
  fig, axes = plt.subplots(1, 2, figsize=(25, 10))
  fig.suptitle('Activity %d played by year'%x, fontsize=20)
  ax = sns.barplot(ax=axes[0], data=year_countx,x='year',y='Activities_played', order=year_countx.sort_values('year').year, color='#00ABC8')
  axes[0].set_title('Activity %d played order by Year'%x, fontsize=20, y=1.05)
  ax.set_xlabel('Year', fontsize=20)
  ax.set_ylabel('Activities played', fontsize=20)
  ax.xaxis.set_tick_params(labelsize='large')
  ax.yaxis.set_tick_params(labelsize='large')
  ax = sns.barplot(ax=axes[1], data=year_countx,x='year',y='Activities_played', order=year_countx.sort_values('Activities_played').year, color='#00ABC8')
  ax.set_xlabel('Year', fontsize=20)
  ax.set_ylabel('Activities played', fontsize=20)
  axes[1].set_title('Activity %d order by activities played'%x, fontsize=20, y=1.05)
  ax.xaxis.set_tick_params(labelsize='large')
  ax.yaxis.set_tick_params(labelsize='large')
  plt.subplots_adjust(wspace=0.3)
 
  return st.pyplot(fig)


#"""## times played by month bar chart function"""

def times_played_month(x):
  activity_x = import_activity(x)

  activity_x['date'] = pd.to_datetime(activity_x['date']).copy()

  activity_x['month']= activity_x['date'].dt.month.copy()

  activity_x['month_name'] = activity_x['date'].dt.month_name().copy()

  by_monthx = activity_x.groupby('month')
  month_countx = by_monthx.aggregate(np.count_nonzero)
  month_countx = month_countx[['patient_id']]
  month_countx = month_countx.reset_index()
  month_countx= month_countx.rename(columns={'patient_id':'Activities_played'})
  
  by_month_namex = activity_x.groupby('month_name')
  month__name_countx = by_month_namex.aggregate(np.count_nonzero)
  month__name_countx = month__name_countx[['patient_id']]
  month__name_countx = month__name_countx.reset_index()
  month__name_countx= month__name_countx.rename(columns={'patient_id':'Activities_played'})

  mean_act_per_monthx = np.mean(month__name_countx['Activities_played'])
  mean_act_per_monthx = str(round(mean_act_per_monthx, 2))
  median_act_per_monthx = np.median(month__name_countx['Activities_played'])
  median_act_per_monthx = str(round(median_act_per_monthx, 2))
  st.write('Mean number of activity %d played per month' % x, mean_act_per_monthx)
  st.write('Median number of activity %d played per month' % x, median_act_per_monthx)
  st.write('\n', month__name_countx.sort_values(by='Activities_played', ascending=False))

  fig, axes = plt.subplots(1, 2, figsize=(25, 10))
  fig.suptitle('Activity %d played by month' % x, fontsize=20)
  ax = sns.barplot(ax=axes[0], data=month__name_countx,x='month_name',y='Activities_played', order=month__name_countx.sort_values('Activities_played').month_name, color='#00ABC8')
  axes[0].set_title('Number of activity %d played per month' %x, fontsize=20, y=1.05)
  ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
  ax.set_xlabel('Month name', fontsize=20)
  ax.set_ylabel('Activities played', fontsize=20)
  ax.xaxis.set_tick_params(labelsize=20)
  ax.yaxis.set_tick_params(labelsize='large')
  ax = sns.barplot(ax=axes[1], data=month_countx,x='month',y='Activities_played', order=month_countx.sort_values('month').month, color='#00ABC8')
  ax.set_xlabel('Month', fontsize=20)
  ax.set_ylabel('Activities played', fontsize=20)
  ax.xaxis.set_tick_params(labelsize=20)
  ax.yaxis.set_tick_params(labelsize='large')
  axes[1].set_title('Number of activity %d played sorted_by_months'% x,fontsize=20, y=1.05)
  plt.subplots_adjust(wspace=0.6)
  return st.pyplot(fig)

#"""## times played by day of week bar chart function"""

def times_played_dayofweek(x):
  activity_x = import_activity(x)

  activity_x['date'] = pd.to_datetime(activity_x['date']).copy()
  activity_x['day_of_week'] = activity_x['date'].dt.dayofweek.copy()
  dw_mapping={
      0: 'Monday',
      1: 'Tuesday', 
      2: 'Wednesday', 
      3: 'Thursday', 
      4: 'Friday',
      5: 'Saturday', 
      6: 'Sunday'} 
  activity_x['day_of_week_name']=activity_x['date'].dt.weekday.map(dw_mapping).copy()

  by_day_of_weekx = activity_x.groupby('day_of_week_name')
  day_of_week_countx = by_day_of_weekx.aggregate(np.count_nonzero)
  day_of_week_countx = day_of_week_countx.rename(columns={'patient_id':'Activities_played'})
  day_of_week_countx = day_of_week_countx[['Activities_played']].sort_values(by='Activities_played', ascending=False)
  st.write(day_of_week_countx)
  day_of_week_countx = day_of_week_countx.reset_index()
  fig, axes = plt.subplots(1, 1, figsize=(20, 10))
  ax = sns.barplot(ax=axes, data=day_of_week_countx,x='day_of_week_name',y='Activities_played', order=day_of_week_countx.sort_values('Activities_played').day_of_week_name, color='#00ABC8')
  ax.set_title('Activity %d played by DayOfWeek'%x, fontsize=20, y=1.05)
  ax.set_xlabel('Day of week', fontsize=20)
  ax.set_ylabel('Activities played', fontsize=20)
  ax.xaxis.set_tick_params(labelsize='large')
  ax.yaxis.set_tick_params(labelsize='large')

  return st.pyplot(fig)

#"""## Show activity result variables function"""

def show_result_variable(x):
  activity_x = import_activity(x)
  result_variables = pd.DataFrame()
  result_variables['Activity %d result variables' % x] = activity_x.columns[9:] 
  return st.write(result_variables)


def show_result_variables(x):
  activity_x = import_activity(x)
  result_variables = pd.DataFrame()
  result_variables['Activity %d result variables' % x] = activity_x.columns[9:]

  
  return (result_variables)

#"""## Show activity domains function"""

def activity_domains(x):
  activity_x = import_activity(x)
  areas_df = pd.read_csv('C:/Users/joshj/INTERNSHIP DATA/All data/areas.csv')
  return st.write(areas_df[areas_df['activity_id']==x])

#"""## Show activity difficulty levels function"""

def difficulty_levels(x):
  activity_x = import_activity(x)
  if activity_type(x) == 'Generator type':
    return st.write('Generator type activities do not have difficulty levels')
  else:
    cross_tab_difficulty = pd.crosstab(index=activity_x['activity_id'],
                             columns=activity_x['difficulty'],
                             normalize="index")
  
    cross_tab_difficulty_df = pd.DataFrame()
    cross_tab_difficulty_df['difficulty'] = cross_tab_difficulty.columns
    cross_tab_difficulty_df['difficulty'] = cross_tab_difficulty_df['difficulty'].round(2)
    
    neuronUP_12colors = ['#00798d','#0089a1','#009ab4','#00abc8','#00bcdc','#00cdef','#04dbff','#66e9ff','#7aecff','#8deeff', '#b4f4ff', '#dcfaff']
    data = cross_tab_difficulty.iloc[0]
    labels = cross_tab_difficulty_df['difficulty']
    colors = sns.color_palette(palette=neuronUP_12colors)
    if len(cross_tab_difficulty.columns) > 6:
      fig, axes = plt.subplots(1, 1, figsize=(20,9))
      fig.suptitle('Proportion of difficulties played in activity %(activity_number)d \n n = %(n)d'% {'activity_number': x, 'n': len(activity_x)},y=1.05, bbox={'facecolor':'0.9', 'pad':8})
      #print('n = %d' % len(activity_x) )
      ax = plt.pie(data, colors=colors, labels=labels, autopct='%.1f%%')
      plt.legend(labels, bbox_to_anchor=(0.9,0.3))
      st.pyplot(fig)
    else:
      fig, axes = plt.subplots(1, 1)
      fig.suptitle('Proportion of difficulties played in activity %(activity_number)d \n n = %(n)d'% {'activity_number': x, 'n': len(activity_x)},y=1.05, bbox={'facecolor':'0.9', 'pad':8})
      ax = plt.pie(data, colors=colors, labels=labels, autopct='%.1f%%')
      plt.legend(labels, bbox_to_anchor=(1,0.4))
      st.pyplot(fig)

  
    return st.write(pd.crosstab(index=activity_x['activity_id'],
                             columns=activity_x['difficulty'].round(2)))
  
  

#"""## Show activity mean scores for each difficulty level function

#"""

def difficulty_scores(x):
  activity_x = import_activity(x)
  if activity_type(x) == 'Generator type':
    return st.write('Generator type activities do not have difficulty levels')
  else:
    by_difficulty = activity_x.groupby('difficulty')
    difficulty_means = by_difficulty.aggregate(np.mean)
    diff_reset = difficulty_means.reset_index()
    diff_reset['difficulty'] = round(diff_reset['difficulty'], 2)
    fig,ax = plt.subplots(1,1)
    ax = sns.barplot(data=diff_reset,x='difficulty',y='score', order=diff_reset.sort_values('difficulty', ascending=True).difficulty, palette=neuronUP_12colors)
    ax.set_title('Average score for each difficulty exercise on activity %d'%x, fontsize=20, y=1.05)
    ax.set_xlabel('Difficulty', fontsize=20)
    ax.set_ylabel('Score mean', fontsize=20)
    ax.xaxis.set_tick_params(labelsize='medium')
    ax.yaxis.set_tick_params(labelsize='large')
    return st.pyplot(fig)

#"""## Show activity mean time to complete for each difficulty level function

#"""

def difficulty_times(x):
  activity_x = import_activity(x)
  if activity_type(x) == 'Game type':
    by_difficulty = activity_x.groupby('difficulty')
    difficulty_means = by_difficulty.aggregate(np.mean)
    diff_reset = difficulty_means.reset_index()
    diff_reset['difficulty'] = round(diff_reset['difficulty'], 2)
    fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    ax = sns.barplot(data=diff_reset,x='difficulty',y='game_pantalla_tiempo', order=diff_reset.sort_values('difficulty', ascending=True).difficulty, palette=neuronUP_12colors)
    ax.set_title('Average time to complete each difficulty exercise on activity %d'%x, fontsize=20, y=1.05)
    ax.set_xlabel('Difficulty', fontsize=20)
    ax.set_ylabel('Game pantalla tiempo', fontsize=20)
    ax.xaxis.set_tick_params(labelsize='large')
    ax.yaxis.set_tick_params(labelsize='large')
    st.pyplot(fig)
    return

  elif activity_type(x) == 'Card type':
    by_difficulty = activity_x.groupby('difficulty')
    difficulty_means = by_difficulty.aggregate(np.mean)
    diff_reset = difficulty_means.reset_index()
    diff_reset['difficulty'] = round(diff_reset['difficulty'], 2)
    fig,ax = plt.subplots(1,1, figsize=(20,10))
    ax = sns.barplot(data=diff_reset,x='difficulty',y='card_tiempo', order=diff_reset.sort_values('difficulty', ascending=True).difficulty, palette=neuronUP_12colors)
    ax.set_title('Average time to complete each difficulty exercise on activity %d'%x, fontsize=20)
    ax.set_xlabel('Difficulty', fontsize=20)
    ax.set_ylabel('Card tiempo', fontsize=20)
    ax.xaxis.set_tick_params(labelsize='large')
    ax.yaxis.set_tick_params(labelsize='large')
    st.pyplot(fig)
    return
  else:
    st.write( 'Generator type activities do not have difficulty levels, this is the average activity time.')
    activity_x1 = activity_x[activity_x['generator_tiempo']<1500]
    fig,ax = plt.subplots(1,1, figsize=(20,10))
    ax = activity_x1['generator_tiempo'].plot(kind='hist', fontsize=14, bins=50, color='#00ABC8')
    ax.set_xlabel('Time taken', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    ax.xaxis.set_tick_params(labelsize='large')
    ax.yaxis.set_tick_params(labelsize='large')
    if activity_x['generator_tiempo'].describe().loc['max'] > 1500:
      plt.title('Average time on activity %d (outliers over 1500s removed)'%x, fontsize=20)
    else:
      plt.title('Average time on activity %d'%x, fontsize=20)
    st.pyplot(fig)    
    return st.write(activity_x['generator_tiempo'].describe())

#"""## Show activity and user scores function

#"""

def scores(x):
  activity_x = import_activity(x)
  by_patientx = activity_x.groupby('patient_id')
  label = 'n = ' + str(len(activity_x)) + '\n '
  patient_score_meanx = by_patientx.aggregate(np.mean)
  patient_score_meanx['score'] = patient_score_meanx['score'].round(2)
  patient_score_meanx_only1 = (patient_score_meanx[patient_score_meanx['score']==1.00])
  fig, axes = plt.subplots(1, 2, figsize=(20,6))
  fig.suptitle('Activity %d: Activity score distribution vs average user scores '%x, fontsize=20, y=1.05)
  ax = activity_x['score'].plot(ax=axes[0], label=label, kind='hist',bins=12, fontsize=20, color='#00ABC8')
  axes[0].set_title('Activity %d score distribution'%x, fontsize=20)
  ax.set_xlabel('Score', fontsize=20)
  ax.set_ylabel('Frequency', fontsize=20)
  ax.xaxis.set_tick_params(labelsize='large')
  ax.yaxis.set_tick_params(labelsize='large')
  ax.legend(loc='best', fontsize=15)
  ax = patient_score_meanx['score'].plot(ax=axes[1], label=label, kind='hist', bins=30, fontsize=14, color='#00ABC8')
  axes[1].set_title('Average Patient score on activity %d'%x, fontsize=20)
  ax.set_xlabel('Score', fontsize=20)
  ax.set_ylabel('Frequency', fontsize=20)
  ax.xaxis.set_tick_params(labelsize='large')
  ax.yaxis.set_tick_params(labelsize='large')
  ax.legend(loc='best', fontsize=15)
  plt.subplots_adjust(wspace=0.5)
  st.pyplot(fig)
  #return st.write(len(patient_score_meanx_only1),"users got an average score of 1")


#"""## Show patient score on certain activity function

#"""

def score_activity_patient(x, y):
  activity_x = import_activity(x)
  by_patientx = activity_x.groupby('patient_id')
  patient_score_meanx = by_patientx.aggregate(np.mean)
  patient_score_meanx = patient_score_meanx.sort_values(by='score', ascending=False)
  patient_score_meanx_reset = patient_score_meanx.reset_index(drop=False)
  patient_y_score = patient_score_meanx_reset[patient_score_meanx_reset['patient_id']==y].loc[:, 'score']

  fig,ax = plt.subplots(1,1)
  ax = patient_score_meanx_reset['score'].plot(kind='hist', fontsize=14, color='#00ABC8', bins=30)
  plt.xlabel('Activity %d - Average patient scores'%x, fontsize=14)
  plt.ylabel('Frequency', fontsize=14)
  plt.plot(patient_y_score, 2 , 'o', markersize = 10,color = 'red', label = 'Patient %d average score'%y)
  plt.title('Activity %(activity_id)d \n Patient %(patient_id)d average score compared to patient population'% {'activity_id': x, 'patient_id': y},y=1.05, fontsize=14)
  ax.legend(loc='best')
  col1, col2 = st.columns(2)
  with col1:
    st.write('n(times_played) = ', len(activity_x))
  with col2:
    st.write('n(patients) = ', len(by_patientx))   
  st.write('patient %d rank out of'%y,len(by_patientx), 'and average score: (0th rank is highest average score)\n', patient_y_score)
  return st.pyplot(fig)

#"""## Show activity time function

#"""

def activity_time(x):
  activity_x = import_activity(x)
  if activity_type(x) == 'Game type':
    activity_x1 = activity_x[activity_x['game_pantalla_tiempo']<1500]
    fig,ax =plt.subplots(1,1)
    ax = activity_x1['game_pantalla_tiempo'].plot(kind='hist', fontsize=14, bins=50, color='#00ABC8')
    plt.xlabel('time taken', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    if activity_x['game_pantalla_tiempo'].describe().loc['max'] > 1500:
        plt.title('Average time on activity %d (outliers over 1500s removed)'%x, fontsize=14)
    else:
        plt.title('Average time on activity %d'%x, fontsize=14)
    st.pyplot(fig)    
    return st.write(activity_x['game_pantalla_tiempo'].describe())

  elif activity_type(x) == 'Card type':
    activity_x1 = activity_x[activity_x['card_tiempo']<1500]
    fig,ax = plt.subplots(1,1)
    ax = activity_x1['card_tiempo'].plot(kind='hist', fontsize=14, bins=50, color='#00ABC8')
    plt.xlabel('time taken', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    if activity_x['card_tiempo'].describe().loc['max'] > 1500:
        plt.title('Average time on activity %d (outliers over 1500s removed)'%x, fontsize=14)
    else:
        plt.title('Average time on activity %d'%x, fontsize=14)
    st.pyplot(fig)    
    return st.write(activity_x['card_tiempo'].describe())

  else:
    activity_x1 = activity_x[activity_x['generator_tiempo']<1500]
    fig,ax = plt.subplots(1,1)
    ax = activity_x1['generator_tiempo'].plot(kind='hist', fontsize=14, bins=50, color='#00ABC8')
    plt.xlabel('time taken', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    if activity_x['generator_tiempo'].describe().loc['max'] > 1500:
        plt.title('Average time on activity %d (outliers over 1500s removed)'%x, fontsize=14)
    else:
        plt.title('Average time on activity %d'%x, fontsize=14)
    st.pyplot(fig)    
    return st.write(activity_x['generator_tiempo'].describe())

#"""## Show activity realizacion function

#"""



def realizacion(x):
  activity_x = import_activity(x)
  
  if activity_type(x) == 'Game type':
    cross_tab_realx = pd.crosstab(index=activity_x['activity_id'],
                             columns=activity_x['game_pantalla_realizacion'],
                             normalize="index")
    data = cross_tab_realx.iloc[0]
    labels = cross_tab_realx.columns
    if np.count_nonzero(cross_tab_realx.columns) <= 2:
      fig, axes = plt.subplots(1, 2, figsize=(15, 5))
      by_realiz = activity_x.groupby('game_pantalla_realizacion')
      realiz_means = by_realiz.aggregate(np.mean)
      realiz_means_sd = realiz_means[['score', 'difficulty']]
      ax = realiz_means_sd.plot(kind='bar', ax=axes[0], color=['#00798d', '#00abc8'], fontsize=12)
      ax.set_ylabel('Mean', fontsize=14)
      axes[0].set_title('Activity %d: Mean score and difficulty by Realizacion'%x, fontsize=14)
      ax.set_xticklabels(ax.get_xticklabels(),rotation = 360, fontsize=14)
      ax.set_xlabel('game_pantalla_realizacion', fontsize=14)
      ax.set_ylabel('Mean', fontsize=14)
      ax.legend(loc='best', fontsize=14)
      ax = plt.pie(data, colors=neuronup_2pie_colors, labels=labels, autopct='%.3f%%')
      plt.title('Proportion of realizacion in activity %d'%x, bbox={'facecolor':'0.9', 'pad':8})
      plt.legend(labels, bbox_to_anchor=(0.9,0.3))
      plt.show()
      st.pyplot(fig)
      return st.write(pd.crosstab(index=activity_x['activity_id'],
                                columns=activity_x['game_pantalla_realizacion']))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        by_realiz = activity_x.groupby('game_pantalla_realizacion')
        realiz_means = by_realiz.aggregate(np.mean)
        realiz_means_sd = realiz_means[['score', 'difficulty']]
        ax = realiz_means_sd.plot(kind='bar', ax=axes[0], color=['#00798d', '#00abc8'], fontsize=12)
        ax.set_ylabel('Mean', fontsize=14)
        axes[0].set_title('Activity %d: Mean score and difficulty by Realizacion'%x, fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(),rotation = 360, fontsize=14)
        ax.set_xlabel('game_pantalla_realizacion', fontsize=14)
        ax.set_ylabel('Mean', fontsize=14)
        ax.legend(loc='best', fontsize=14)


        ax = plt.pie(data, colors=neuronup_pie_colors, labels=labels, autopct='%.1f%%')
        plt.title('Proportion of realizacion in activity %d'%x, bbox={'facecolor':'0.9', 'pad':8})
        plt.legend(labels, bbox_to_anchor=(0.9,0.3))
        plt.show()
        st.pyplot(fig)
        return st.write(pd.crosstab(index=activity_x['activity_id'],
                                  columns=activity_x['game_pantalla_realizacion']))
            
  elif activity_type(x) == 'Card type':
    cross_tab_realx = pd.crosstab(index=activity_x['activity_id'],
                             columns=activity_x['card_realizacion'],
                             normalize="index")
    data = cross_tab_realx.iloc[0]
    labels = cross_tab_realx.columns
    if np.count_nonzero(cross_tab_realx.columns) <= 2:
      fig, axes = plt.subplots(1, 2, figsize=(15, 5))

      by_realiz = activity_x.groupby('card_realizacion')
      realiz_means = by_realiz.aggregate(np.mean)
      realiz_means_sd = realiz_means[['score', 'difficulty']]
      ax = realiz_means_sd.plot(kind='bar', ax=axes[0], color=['#00798d', '#00abc8'], fontsize=12)
      ax.set_ylabel('Mean', fontsize=14)
      axes[0].set_title('Activity %d: Mean score and difficulty by Realizacion'%x, fontsize=14)
      ax.set_xticklabels(ax.get_xticklabels(),rotation = 360, fontsize=14)
      ax.set_xlabel('card_realizacion', fontsize=14)
      ax.set_ylabel('Mean', fontsize=14)
      ax.legend(loc='best', fontsize=14)


      ax = plt.pie(data, colors=neuronup_2pie_colors, labels=labels, autopct='%.3f%%')
      plt.title('Proportion of realizacion in activity %d'%x, bbox={'facecolor':'0.9', 'pad':8})
      plt.legend(labels, bbox_to_anchor=(0.9,0.3))
      plt.show()
      st.pyplot(fig)
      return st.write(pd.crosstab(index=activity_x['activity_id'],
                                columns=activity_x['card_realizacion']))
    else:
      fig, axes = plt.subplots(1, 2, figsize=(15, 5))

      by_realiz = activity_x.groupby('card_realizacion')
      realiz_means = by_realiz.aggregate(np.mean)
      realiz_means_sd = realiz_means[['score', 'difficulty']]
      ax = realiz_means_sd.plot(kind='bar', ax=axes[0], color=['#00798d', '#00abc8'], fontsize=12)
      ax.set_ylabel('Mean', fontsize=14)
      axes[0].set_title('Activity %d: Mean score and difficulty by Realizacion'%x, fontsize=14)
      ax.set_xticklabels(ax.get_xticklabels(),rotation = 360, fontsize=14)
      ax.set_xlabel('card_realizacion', fontsize=14)
      ax.set_ylabel('Mean', fontsize=14)
      ax.legend(loc='best', fontsize=14)
      ax = plt.pie(data, colors=neuronup_pie_colors, labels=labels, autopct='%.1f%%')
      plt.title('Proportion of realizacion in activity %d'%x, bbox={'facecolor':'0.9', 'pad':8})
      plt.legend(labels, bbox_to_anchor=(0.9,0.3))
      plt.show()
      st.pyplot(fig)
      return st.write(pd.crosstab(index=activity_x['activity_id'],
                                  columns=activity_x['card_realizacion']))

  else:
    cross_tab_realx = pd.crosstab(index=activity_x['activity_id'],
                             columns=activity_x['generator_realizacion'],
                             normalize="index")
    data = cross_tab_realx.iloc[0]
    labels = cross_tab_realx.columns
    print('Generator activities do not have different difficulties')
    if np.count_nonzero(cross_tab_realx.columns) <= 2:
       fig, axes = plt.subplots(1, 2, figsize=(15, 5))
       by_realiz = activity_x.groupby('generator_realizacion')
       realiz_means = by_realiz.aggregate(np.mean)
       realiz_means_sd = realiz_means[['score', 'difficulty']]
       ax = realiz_means_sd.plot(kind='bar', ax=axes[0], color=['#00798d', '#00abc8'], fontsize=12)
       ax.set_ylabel('Mean', fontsize=14)
       axes[0].set_title('Activity %d: Mean score and difficulty by Realizacion'%x, fontsize=14)
       ax.set_xticklabels(ax.get_xticklabels(),rotation = 360, fontsize=14)
       ax.set_xlabel('generator_realizacion', fontsize=14)
       ax.set_ylabel('Mean', fontsize=14)
       ax.legend(loc='best', fontsize=14)
       ax = plt.pie(data, colors=neuronup_2pie_colors, labels=labels, autopct='%.3f%%')
       plt.title('Proportion of realizacion in activity %d'%x, bbox={'facecolor':'0.9', 'pad':8})
       plt.legend(labels, bbox_to_anchor=(0.9,0.3))
       plt.show()
       st.pyplot(fig)
       return st.write(pd.crosstab(index=activity_x['activity_id'],
                                    columns=activity_x['generator_realizacion']))
    else:
         fig, axes = plt.subplots(1, 2, figsize=(15, 5))
         by_realiz = activity_x.groupby('generator_realizacion')
         realiz_means = by_realiz.aggregate(np.mean)
         realiz_means_sd = realiz_means[['score', 'difficulty']]
         ax = realiz_means_sd.plot(kind='bar', ax=axes[0], color=['#00798d', '#00abc8'], fontsize=12)
         ax.set_ylabel('Mean', fontsize=14)
         axes[0].set_title('Activity %d: Mean score and difficulty by Realizacion'%x, fontsize=14)
         ax.set_xticklabels(ax.get_xticklabels(),rotation = 360, fontsize=14)
         ax.set_xlabel('generator_realizacion', fontsize=14)
         ax.set_ylabel('Mean', fontsize=14)
         ax.legend(loc='best',  fontsize=14)
         ax = plt.pie(data, colors=neuronup_pie_colors, labels=labels, autopct='%.1f%%')
         plt.title('Proportion of realizacion in activity %d'%x, bbox={'facecolor':'0.9', 'pad':8})
         plt.legend(labels, bbox_to_anchor=(0.9,0.3))
         plt.show()
         st.pyplot(fig)
         return st.write(pd.crosstab(index=activity_x['activity_id'],
                                      columns=activity_x['generator_realizacion']))



#"""## Show activity intentos function

#"""

def intentos(x):
  activity_x = import_activity(x)
  varx = (show_result_variables(x))
  boolean = varx['Activity %d result variables'%x].str.contains('intentos')
  if np.count_nonzero(boolean) == 0:
    return st.write('This activity does not have an intentos variable')
  else:
    if activity_type(x) == 'Game type':
      by_difficulty = activity_x.groupby('difficulty')
      difficulty_means = by_difficulty.aggregate(np.mean)
      diff_reset = difficulty_means.reset_index()
      diff_reset['difficulty'] = round(diff_reset['difficulty'], 2)
      fig,ax = plt.subplots(1,1)
      ax = sns.barplot(data=diff_reset,x='difficulty',y='game_intentos', order=diff_reset.sort_values('difficulty', ascending=True).difficulty, palette=neuronUP_10colors_darktolight)
      ax.set_title('Average intentos for each difficulty exercise on activity %d'%x, fontsize=14)
      ax.set_ylabel('Average game intentos')
      return st.pyplot(fig)
    elif activity_type(x) == 'Card type':
      by_difficulty = activity_x.groupby('difficulty')
      difficulty_means = by_difficulty.aggregate(np.mean)
      diff_reset = difficulty_means.reset_index()
      diff_reset['difficulty'] = round(diff_reset['difficulty'], 2)
      fig,ax = plt.subplots(1,1)
      ax = sns.barplot(data=diff_reset,x='difficulty',y='card_intentos', order=diff_reset.sort_values('difficulty', ascending=True).difficulty, palette=neuronUP_10colors_darktolight)
      ax.set_title('Average intentos for each difficulty exercise on activity %d'%x, fontsize=14)
      ax.set_ylabel('Average card intentos')
      return st.pyplot(fig)
    else:
      st.write('Generator types do not have difficulty levels, here is average attempts for different score categories')
      st.write('*Note, high score is over 0.8, low score is under 0.4')
      activity_x_high_score = activity_x[activity_x['score']>0.8]
      activity_x_high_score_intentos_mean = np.mean(activity_x_high_score['generator_intentos'])
      activity_x_low_score = activity_x[activity_x['score']<0.4]
      activity_x_low_score_intentos_mean = np.mean(activity_x_low_score['generator_intentos'])
      higher_04 = activity_x[activity_x['score']>0.4]
      activity_x_middle_score = higher_04[higher_04['score'] < 0.8]
      activity_x_middle_score_intentos_mean = np.mean(activity_x_middle_score['generator_intentos'])
      x = ['high_score', 'middle_score', 'low_score']
      y = [activity_x_high_score_intentos_mean, activity_x_middle_score_intentos_mean, activity_x_low_score_intentos_mean]
      fig,ax = plt.subplots(1,1)
      ax = plt.bar(x, y, color='#00ABC8')
      plt.title("average Intentos for different score level", fontsize=14)
      plt.ylabel('Average intentos', fontsize=14)
      return st.pyplot(fig)

#"""## Show activity success v errors function

#"""

def aciertos_errores(x):
  activity_x = import_activity(x)
  varx = show_result_variables(x)
  boolean_aciertos = varx['Activity %d result variables'%x].str.contains('aciertos')
  boolean_errores = varx['Activity %d result variables'%x].str.contains('errores')
  if (np.count_nonzero(boolean_aciertos)) + (np.count_nonzero(boolean_errores)) != 2:
    return st.write('This activity does not have both an aciertos and errores variable')

  else:

    if activity_type(x) == 'Game type':
      activity_x['proportion_aciertos'] = activity_x['game_aciertos'] / (activity_x['game_aciertos'] + activity_x['game_errores'])
      fig,ax = plt.subplots(1,1)
      ax = activity_x['proportion_aciertos'].plot(kind='hist', fontsize=14, color='#00ABC8')
      plt.xlabel('Proportion of aciertos', fontsize=14)
      plt.title('Activity %d: Proportion of aciertos compared to errores'%x,y=1.05, fontsize=14)
      st.pyplot(fig)
      return st.write(activity_x['proportion_aciertos'].describe())


    elif activity_type(x) == 'Card type':
      activity_x['proportion_aciertos'] = activity_x['card_aciertos'] / (activity_x['card_aciertos'] + activity_x['card_errores'])
      fig,ax = plt.subplots(1,1)
      ax = activity_x['proportion_aciertos'].plot(kind='hist', fontsize=14, color='#00ABC8')
      plt.xlabel('Proportion of aciertos', fontsize=14)
      plt.title('Activity %d: Proportion of aciertos compared to errores'%x, y=1.05,fontsize=14)
      st.pyplot(fig)
      return st.write(activity_x['proportion_aciertos'].describe())

    else:
      activity_x['proportion_aciertos'] = activity_x['generator_aciertos'] / (activity_x['generator_aciertos'] + activity_x['generator_errores'])
      fig,ax = plt.subplots(1,1)
      ax = activity_x['proportion_aciertos'].plot(kind='hist', fontsize=14, color='#00ABC8')
      plt.xlabel('Proportion of aciertos', fontsize=14)
      plt.title('Activity %d: Proportion of aciertos compared to errores'%x,y=1.05, fontsize=14)
      st.pyplot(fig)
      return st.write(activity_x['proportion_aciertos'].describe())

#"""## Show game activity sublevels function

#"""

def game_fase_pantalla_subenivel(x):
  activity_x = import_activity(x)
  varx = show_result_variables(x)
  boolean_sub = varx['Activity %d result variables'%x].str.contains('subenivel')
  if np.count_nonzero(boolean_sub) == 0:
    return st.write('This activity does not have sublevels')
  else:
    cross_tab_sublevel = pd.crosstab(index=activity_x['activity_id'],
                             columns=activity_x['game_fase_pantalla_subenivel'])
    by_subenivel = activity_x.groupby('game_fase_pantalla_subenivel')
    subenivel_means = by_subenivel.aggregate(np.mean)
    subenivel_means_sd = subenivel_means[['score', 'difficulty']]
    
    fig, axes = plt.subplots(1, 1)
    ax = subenivel_means_sd.plot(kind='bar', ax=axes, color=['#00798d', '#00abc8'], fontsize=12)
    ax.set_ylabel('Mean', fontsize=14)
    ax.set_title('\n Activity %d: Mean score and difficulty by subenivel'%x, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 360, fontsize=14)
    ax.set_xlabel('Sublevel', fontsize=14)
    ax.legend(loc='best', bbox_to_anchor=(0.9,0.3), fontsize=14)
    st.pyplot(fig)
    return st.write(cross_tab_sublevel)

#['Choose visualisation', 'Activity types', 'Domain counts(parent name, sub name)','Parent name mean scores', 'Most played activities',
#    'Least played activities', 'Activities with best score', 'Activities with worst score', 'Times played by month', 'times played by day of week']

def year_activity_types(year):
  year_x = import_data(year)
  game_card_gen = year_x['type'].value_counts()
  by_type = year_x.groupby('type')
  type_means = by_type.aggregate(np.mean)
  mean_score_byType = type_means[['score', 'difficulty']]
  st.write('Number of game types:', game_card_gen[0],  '      Mean game score:', type_means.loc['game','score'] )
  st.write('Number of card types:', game_card_gen[1],  '      Mean card score:', type_means.loc['card','score'])
  st.write('Number of generator types:', game_card_gen[2],  '  Mean generator score:', type_means.loc['generator','score'])
  mean_score_byType_reset = mean_score_byType.reset_index()
  fig, axes = plt.subplots(1, 1)
  ax = mean_score_byType.plot(ax=axes, kind='bar', color=['#00798d', '#00abc8'], fontsize=12)
  ax.set_title('Mean score and difficulty by type', fontsize=14)
  ax.set_ylabel('Mean', fontsize=20)
  st.pyplot(fig)

def year_domain_counts(year):
  year_x = import_data(year)
  url = 'https://drive.google.com/file/d/1bXMBe74hu05XtVnsjLzyX08-ZjJ2s0Od/view?usp=sharing'
  path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
  areas_df = pd.read_csv('C:/Users/joshj/INTERNSHIP DATA/All data/areas.csv')
  parent_counts = (areas_df['parent_name'].value_counts())
  name_counts = areas_df['name'].value_counts().head(20)
  parent_df = pd.DataFrame()
  parent_df['counts'] = parent_counts
  parent_reset = parent_df.reset_index()
  parent_reset = parent_reset.rename(columns={"index": "parent_name"})

  name_df = pd.DataFrame()
  name_df['counts'] = name_counts
  name_reset = name_df.reset_index()
  name_reset = name_reset.rename(columns={"index": "sub_name"})

  plt.style.use('seaborn')

  
  x1 = parent_reset['parent_name']
  x2 = name_reset['sub_name']
  y1 = parent_reset['counts']
  y2 = name_reset['counts']

  fig, axes = plt.subplots(1, 2, figsize=(15, 8))
  fig.suptitle('most played parent and sub domains', fontsize=20, y=1.05)
  ax = sns.barplot(ax=axes[0], x=y1, y=x1, order=parent_reset.sort_values('counts').parent_name,orient='h', color='#00ABC8')
  axes[0].set_title('Parent name counts',fontsize=20)
  ax.set_xlabel('Counts', fontsize=20)
  ax.set_ylabel('Parent name', fontsize=20)
  ax.xaxis.set_tick_params(labelsize='large')
  ax.yaxis.set_tick_params(labelsize='large')

  ax = sns.barplot(ax=axes[1], x=y2, y=x2, order=name_reset.sort_values('counts').sub_name, orient='h',color='#00ABC8')
  axes[1].set_title('Sub-name counts',fontsize=20)
  ax.set_xlabel('Counts', fontsize=20)
  ax.set_ylabel('Sub-name', fontsize=20)
  ax.xaxis.set_tick_params(labelsize='large')
  ax.yaxis.set_tick_params(labelsize='large')
  plt.subplots_adjust(left=0, right=1, wspace=0.6)
  st.pyplot(fig)

def year_parent_means(year):
    year_x = import_data(year)
    url = 'https://drive.google.com/file/d/1bXMBe74hu05XtVnsjLzyX08-ZjJ2s0Od/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    areas_df = pd.read_csv('C:/Users/joshj/INTERNSHIP DATA/All data/areas.csv')
    parent_name_values = areas_df['parent_name'].value_counts()
    parent_value_df = pd.DataFrame()
    parent_value_df['parent_name'] = parent_name_values
    parent_value_df_reset = parent_value_df.reset_index()
    parent_value_df = parent_value_df_reset.rename(columns={"index": "parent_name", "parent_name": "counts"})
    parent_value_df = parent_value_df.head(12)

    attention_only = areas_df[areas_df['parent_name']=='Attention']
    # root_only = areas_df[areas_df['parent_name']=='Root']
    executive_func_only = areas_df[areas_df['parent_name']=='Executive Functions']
    memory_only = areas_df[areas_df['parent_name']=='Memory']
    gnosis_only = areas_df[areas_df['parent_name']=='Gnosis']
    language_only = areas_df[areas_df['parent_name']=='Language']
    visuospatial_only = areas_df[areas_df['parent_name']=='Visuospatial Skills']
    cogfunc_only = areas_df[areas_df['parent_name']=='Cognitive Functions']
    instrumentalADL_only = areas_df[areas_df['parent_name']=="Instrumental ADL's"]
    praxis_only = areas_df[areas_df['parent_name']=='Praxis']
    basicADL_only = areas_df[areas_df['parent_name']=="Basic ADL's"]
    orientation_only = areas_df[areas_df['parent_name']=='Orientation']
    occupation_only = areas_df[areas_df['parent_name']=='Areas of Occupation']
    attention_activityIDs = attention_only['activity_id'].value_counts()
    attention_id_df = pd.DataFrame()
    attention_id_df['attention_IDs'] = attention_activityIDs
    attention_id_df_reset = attention_id_df.reset_index()
    attention_id_df = attention_id_df_reset.rename(columns={"index": "attention_ids", "attention_IDs": "counts"})
    attention_IDs = attention_id_df['attention_ids']
    attention_only_data  = year_x[year_x['activity_id'].isin(attention_IDs)]
    attention_activities_score_mean = attention_only_data['score'].mean()
    
    #root_activityIDs = root_only['activity_id'].value_counts()
    #root_id_df = pd.DataFrame()
    #root_id_df['root_IDs'] = root_activityIDs
    #root_id_df_reset = root_id_df.reset_index()
    #root_id_df = root_id_df_reset.rename(columns={"index": "root_ids", "root_IDs": "counts"})
    #root_IDs = root_id_df['root_ids']
    #root_only_data  = year_x[year_x['activity_id'].isin(root_IDs)]
    #root_activities_score_mean = root_only_data['score'].mean()

    executive_func_activityIDs = executive_func_only['activity_id'].value_counts()
    executive_func_id_df = pd.DataFrame()
    executive_func_id_df['executive_func_IDs'] = executive_func_activityIDs
    executive_func_id_df_reset = executive_func_id_df.reset_index()
    executive_func_id_df = executive_func_id_df_reset.rename(columns={"index": "executive_func_ids", "executive_func_IDs": "counts"})
    executive_func_IDs = executive_func_id_df['executive_func_ids']
    executive_func_only_data  = year_x[year_x['activity_id'].isin(executive_func_IDs)]
    executive_func_activities_score_mean = executive_func_only_data['score'].mean()
    memory_activityIDs = memory_only['activity_id'].value_counts()
    memory_id_df = pd.DataFrame()
    memory_id_df['memory_IDs'] = memory_activityIDs
    memory_id_df_reset = memory_id_df.reset_index()
    memory_id_df = memory_id_df_reset.rename(columns={"index": "memory_ids", "memory_IDs": "counts"})
    memory_IDs = memory_id_df['memory_ids']
    memory_only_data  = year_x[year_x['activity_id'].isin(memory_IDs)]
    memory_activities_score_mean = memory_only_data['score'].mean()
    gnosis_activityIDs = gnosis_only['activity_id'].value_counts()
    gnosis_id_df = pd.DataFrame()
    gnosis_id_df['gnosis_IDs'] = gnosis_activityIDs
    gnosis_id_df_reset = gnosis_id_df.reset_index()
    gnosis_id_df = gnosis_id_df_reset.rename(columns={"index": "gnosis_ids", "gnosis_IDs": "counts"})
    gnosis_IDs = gnosis_id_df['gnosis_ids']
    gnosis_only_data  = year_x[year_x['activity_id'].isin(gnosis_IDs)]
    gnosis_activities_score_mean = gnosis_only_data['score'].mean()
    language_activityIDs = language_only['activity_id'].value_counts()
    language_id_df = pd.DataFrame()
    language_id_df['language_IDs'] = language_activityIDs
    language_id_df_reset = language_id_df.reset_index()
    language_id_df = language_id_df_reset.rename(columns={"index": "language_ids", "language_IDs": "counts"})
    language_IDs = language_id_df['language_ids']
    language_only_data  = year_x[year_x['activity_id'].isin(language_IDs)]
    language_activities_score_mean = language_only_data['score'].mean()
    visuospatial_activityIDs = visuospatial_only['activity_id'].value_counts()
    visuospatial_id_df = pd.DataFrame()
    visuospatial_id_df['visuospatial_IDs'] = visuospatial_activityIDs
    visuospatial_id_df_reset = visuospatial_id_df.reset_index()
    visuospatial_id_df = visuospatial_id_df_reset.rename(columns={"index": "visuospatial_ids", "visuospatial_IDs": "counts"})
    visuospatial_IDs = visuospatial_id_df['visuospatial_ids']
    visuospatial_only_data  = year_x[year_x['activity_id'].isin(visuospatial_IDs)]
    visuospatial_activities_score_mean = visuospatial_only_data['score'].mean()
    cogfunc_activityIDs = cogfunc_only['activity_id'].value_counts()
    cogfunc_id_df = pd.DataFrame()
    cogfunc_id_df['cogfunc_IDs'] = cogfunc_activityIDs
    cogfunc_id_df_reset = cogfunc_id_df.reset_index()
    cogfunc_id_df = cogfunc_id_df_reset.rename(columns={"index": "cogfunc_ids", "cogfunc_IDs": "counts"})
    cogfunc_IDs = cogfunc_id_df['cogfunc_ids']
    cogfunc_only_data  = year_x[year_x['activity_id'].isin(cogfunc_IDs)]
    cogfunc_activities_score_mean = cogfunc_only_data['score'].mean()
    instrumentalADL_activityIDs = instrumentalADL_only['activity_id'].value_counts()
    instrumentalADL_id_df = pd.DataFrame()
    instrumentalADL_id_df['instrumentalADL_IDs'] = instrumentalADL_activityIDs
    instrumentalADL_id_df_reset = instrumentalADL_id_df.reset_index()
    instrumentalADL_id_df = instrumentalADL_id_df_reset.rename(columns={"index": "instrumentalADL_ids", "instrumentalADL_IDs": "counts"})
    instrumentalADL_IDs = instrumentalADL_id_df['instrumentalADL_ids']
    instrumentalADL_only_data  = year_x[year_x['activity_id'].isin(instrumentalADL_IDs)]
    instrumentalADL_activities_score_mean = instrumentalADL_only_data['score'].mean()
    praxis_activityIDs = praxis_only['activity_id'].value_counts()
    praxis_id_df = pd.DataFrame()
    praxis_id_df['praxis_IDs'] = praxis_activityIDs
    praxis_id_df_reset = praxis_id_df.reset_index()
    praxis_id_df = praxis_id_df_reset.rename(columns={"index": "praxis_ids", "praxis_IDs": "counts"})
    praxis_IDs = praxis_id_df['praxis_ids']
    praxis_only_data  = year_x[year_x['activity_id'].isin(praxis_IDs)]
    praxis_activities_score_mean = praxis_only_data['score'].mean()
    basicADL_activityIDs = basicADL_only['activity_id'].value_counts()
    basicADL_id_df = pd.DataFrame()
    basicADL_id_df['basicADL_IDs'] = basicADL_activityIDs
    basicADL_id_df_reset = basicADL_id_df.reset_index()
    basicADL_id_df = basicADL_id_df_reset.rename(columns={"index": "basicADL_ids", "basicADL_IDs": "counts"})
    basicADL_IDs = basicADL_id_df['basicADL_ids']
    basicADL_only_data  = year_x[year_x['activity_id'].isin(basicADL_IDs)]
    basicADL_activities_score_mean = basicADL_only_data['score'].mean()
    orientation_activityIDs = orientation_only['activity_id'].value_counts()
    orientation_id_df = pd.DataFrame()
    orientation_id_df['orientation_IDs'] = orientation_activityIDs
    orientation_id_df_reset = orientation_id_df.reset_index()
    orientation_id_df = orientation_id_df_reset.rename(columns={"index": "orientation_ids", "orientation_IDs": "counts"})
    orientation_IDs = orientation_id_df['orientation_ids']
    orientation_only_data  = year_x[year_x['activity_id'].isin(orientation_IDs)]
    orientation_activities_score_mean = orientation_only_data['score'].mean()
    occupation_activityIDs = occupation_only['activity_id'].value_counts()
    occupation_id_df = pd.DataFrame()
    occupation_id_df['occupation_IDs'] = occupation_activityIDs
    occupation_id_df_reset = occupation_id_df.reset_index()
    occupation_id_df = occupation_id_df_reset.rename(columns={"index": "occupation_ids", "occupation_IDs": "counts"})
    occupation_IDs = occupation_id_df['occupation_ids']
    occupation_only_data  = year_x[year_x['activity_id'].isin(occupation_IDs)]
    occupation_activities_score_mean = occupation_only_data['score'].mean()
    means = print(attention_activities_score_mean, executive_func_activities_score_mean, instrumentalADL_activities_score_mean, language_activities_score_mean, memory_activities_score_mean, basicADL_activities_score_mean,
                  occupation_activities_score_mean, gnosis_activities_score_mean, cogfunc_activities_score_mean, visuospatial_activities_score_mean, 
                  praxis_activities_score_mean, orientation_activities_score_mean)
    means = np.array([attention_activities_score_mean, executive_func_activities_score_mean, instrumentalADL_activities_score_mean, language_activities_score_mean, memory_activities_score_mean, basicADL_activities_score_mean,
                  occupation_activities_score_mean, gnosis_activities_score_mean, cogfunc_activities_score_mean, visuospatial_activities_score_mean, 
                  praxis_activities_score_mean, orientation_activities_score_mean])
    parent_name_means_df = pd.DataFrame()
    parent_name_means_df['parent_name'] = parent_value_df['parent_name']
    parent_name_means_df['score_mean'] = means
    fig, axes = plt.subplots(1, 1)
    ax = sns.barplot(ax=axes, data=parent_name_means_df,x='parent_name',y='score_mean', order=parent_name_means_df.sort_values('score_mean').parent_name, palette=neuronUP_12colors)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    ax.set_xlabel('Parent name', fontsize=20)
    ax.set_ylabel('Score Mean', fontsize=20)
    ax.set_title('Mean score for each parent name', fontsize=20)
    st.pyplot(fig)

def year_most_played_activities(year):
  year_x = import_data(year)
  Top10_activities = pd.DataFrame()
  Top10_activities['times_played'] = year_x['activity_id'].value_counts().head(10)
  top10_reset = Top10_activities.reset_index()
  top10_rename = top10_reset.rename(columns={'index':'activity_id'})
  fig, axes = plt.subplots(1, 1)
  ax = sns.barplot(ax=axes, data=top10_rename,x='activity_id',y='times_played', order=top10_rename.sort_values('times_played').activity_id, palette=neuronUP_10colors)
  ax.set_title('Most Played Activity IDs', fontsize=20)
  st.pyplot(fig)

def year_least_played_activities(year):
  year_x = import_data(year)
  bottom10_activities = pd.DataFrame()
  bottom10_activities['times_played'] = year_x['activity_id'].value_counts().sort_values(ascending=True).head(10)
  bottom10_reset = bottom10_activities.reset_index()
  bottom10_rename = bottom10_reset.rename(columns={'index':'activity_id'})
  fig, axes = plt.subplots(1, 1)
  ax = sns.barplot(ax=axes, data=bottom10_rename,x='activity_id',y='times_played', order=bottom10_rename.sort_values('times_played', ascending=False).activity_id, palette=neuronUP_10colors_darktolight)
  ax.set_title('Least Played Activity IDs', fontsize=20)
  st.pyplot(fig)

def year_best_activities(year):
  year_x = import_data(year)
  by_act_type = year_x.groupby('activity_id')
  type_act_means = by_act_type.aggregate(np.mean).sort_values(by='score', ascending=False)
  best_scores = type_act_means[['score']].head(100)

  return st.dataframe(best_scores)

def year_worst_activities(year):
  year_x = import_data(year)
  by_activity = year_x.groupby('activity_id')
  type_activity_means = by_activity.aggregate(np.mean).sort_values(by='score', ascending=True)
  worst_scores= type_activity_means[['score']].head(100)
  return st.dataframe(worst_scores)

def year_activities_per_month(year):
  year_x = import_data(year)
  date_activities = year_x[['activity_id', 'date', 'difficulty', 'score']]
  st.write('Number of activities played in %d = '%year, len(date_activities))
  date_activities['date'] = pd.to_datetime(date_activities['date'])
  date_activities['month']= date_activities['date'].dt.month
  date_activities['month_name'] = date_activities['date'].dt.month_name()
  by_month = date_activities.groupby('month')
  month_count = by_month.aggregate(np.count_nonzero)
  month_count = month_count[['activity_id']]
  month_count = month_count.reset_index()
  month_count= month_count.rename(columns={'activity_id':'Activities_played'})
  by_month_name = date_activities.groupby('month_name')
  month__name_count = by_month_name.aggregate(np.count_nonzero)
  month__name_count = month__name_count[['activity_id']]
  month__name_count = month__name_count.reset_index()
  month__name_count= month__name_count.rename(columns={'activity_id':'Activities_played'})
  mean_act_per_month = np.mean(month__name_count['Activities_played'])
  mean_act_per_month = str(round(mean_act_per_month, 2))
  median_act_per_month = np.median(month__name_count['Activities_played'])
  median_act_per_month = str(round(median_act_per_month, 2))
  st.write('Mean number of activities played per month:', mean_act_per_month)
  st.write('Median number of activities played per month:', median_act_per_month)
  fig, axes = plt.subplots(1, 2, figsize=(20, 6))
  fig.suptitle('Activities played by month', fontsize=20, y=1.05)
  ax = sns.barplot(ax=axes[0], data=month__name_count,x='month_name',y='Activities_played', order=month__name_count.sort_values('Activities_played').month_name, color='#00ABC8')
  axes[0].set_title('Number of activities played every month in %d'%year, fontsize=20, y=1.05)
  ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
  ax = sns.barplot(ax=axes[1], data=month_count,x='month',y='Activities_played', order=month_count.sort_values('month').month, color='#00ABC8')
  axes[1].set_title('Number of activities played %d sorted_by_months'%year, y=1.05)
  plt.subplots_adjust(wspace=0.6)
  st.pyplot(fig)
  return st.dataframe(month__name_count)

def year_activities_day_of_week(year):
  year_x = import_data(year)
  date_activities = year_x[['activity_id', 'date', 'difficulty', 'score']]
  st.write('Number of activities played in %d = '%year, len(date_activities))
  date_activities['date'] = pd.to_datetime(date_activities['date'])
  date_activities['week_of_year'] = date_activities['date'].dt.isocalendar().week
  date_activities['day_of_week'] = date_activities['date'].dt.dayofweek
  dw_mapping={
      0: 'Monday',
      1: 'Tuesday', 
      2: 'Wednesday', 
      3: 'Thursday', 
      4: 'Friday',
      5: 'Saturday', 
      6: 'Sunday'} 
  date_activities['day_of_week_name']=date_activities['date'].dt.weekday.map(dw_mapping)
  by_week = date_activities.groupby('week_of_year')
  week_count = by_week.aggregate(np.count_nonzero)
  week_count = week_count.rename(columns={'activity_id':'Activities_played'})
  mean_act_per_week = np.mean(week_count['Activities_played'])
  mean_act_per_week = str(round(mean_act_per_week, 2))
  median_act_per_week = np.median(week_count['Activities_played'])
  median_act_per_week = str(round(median_act_per_week, 2))  
  st.write('Mean number of activities played per week:', mean_act_per_week)
  st.write('Median number of activities played per week:', median_act_per_week)
  by_day_of_week = date_activities.groupby('day_of_week_name')
  day_of_week_count = by_day_of_week.aggregate(np.count_nonzero)
  day_of_week_count = day_of_week_count.rename(columns={'activity_id':'Activities_played'})
  day_of_week_count = day_of_week_count[['Activities_played']].sort_values(by='Activities_played', ascending=False)
  st.dataframe(day_of_week_count)
  day_of_week_count = day_of_week_count.reset_index()
  fig, axes = plt.subplots(1, 1, figsize=(20, 10))
  ax = sns.barplot(ax=axes, data=day_of_week_count,x='day_of_week_name',y='Activities_played', order=day_of_week_count.sort_values('Activities_played').day_of_week_name, color='#00ABC8')
  ax.set_title('Activities played by day of week in year %d'%year, fontsize=20, y=1.05)
  ax.set_xlabel('Day of week', fontsize=20)
  ax.set_ylabel('Activities played', fontsize=20)
  ax.xaxis.set_tick_params(labelsize='large')
  ax.yaxis.set_tick_params(labelsize='large')

  return st.pyplot(fig)
# from example, to get NeuronUP logo and color

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 325px;
        background-color: rgb(240,250,250);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<div style="text-align: left;"><img src="data:image/svg+xml;base64,%s"/></div>' % b64
    st.sidebar.write(html, unsafe_allow_html=True)
url = "https://www.neuronup.com/wp-content/uploads/2021/07/logo-neuronup-core.svg"
r = requests.get(url) # Get the webpage
svg = r.content.decode() # Decoded response content with the svg string
render_svg(svg) # Render the svg string
st.sidebar.markdown("<h1 style='text-align: left; font-size: 35px;\
color:#636569;'>Hello Streamlit</h1>", unsafe_allow_html=True)


st.sidebar.markdown("#### Data analysis visualisation")

choose_option = st.sidebar.selectbox(
    'Choose one option',
    ['Option', 'Global activity analysis (visuals take longer to load)', 'Specific activity analysis'])
if choose_option == 'Option':
   st.title('')
   st.write('#### Please choose an analysis option')
   st.write('')
   st.write('###### * Note that Global activity analysis visualisations can take up to a minute to load due to millions of activities played')
   st.write('###### * Specific activity analysis loads very quickly')
elif choose_option == 'Global activity analysis (visuals take longer to load)':
  choose_year = st.sidebar.selectbox(
    'Choose year \n (* note: 2018 is all data upto and including 2018)',
    ['Choose year','2018','2019', '2020','2021', '2022'])
  if choose_year == 'Choose year':
    'Please choose a year you wish to analyse the data from'
  else:
    choose_visualisation = st.sidebar.selectbox(
    'Choose a visualisation of year: ' + choose_year,
    ['Choose visualisation', 'Activity types', 'Domain counts(parent name, sub name)','Parent name mean scores', 'Most played activities',
    'Least played activities', 'Activities with best average scores', 'Activities with worst average scores', 'Times played by month', 'Times played by day of week'])
    if choose_visualisation == 'Choose visualisation':
      'Please choose a visualisation option'
    elif choose_visualisation == 'Activity types':
      st.header('Activity types average score and difficulties for year %d'%int(choose_year))  
      year_activity_types(int(choose_year))
      
    elif choose_visualisation == 'Domain counts(parent name, sub name)':
      st.header('Parent and sub-name counts in year: %d'%int(choose_year))  
      year_domain_counts(int(choose_year))
      
    elif choose_visualisation == 'Parent name mean scores':
      st.header('Parent names mean scores in year: %d'%int(choose_year))  
      year_parent_means(int(choose_year))
      
    elif choose_visualisation == 'Most played activities':
      st.header('Most played activities in year: %d'%int(choose_year))  
      year_most_played_activities(int(choose_year))
      
    elif choose_visualisation == 'Least played activities':
      st.header('Least played activities in year: %d'%int(choose_year))  
      year_least_played_activities(int(choose_year))
      
    elif choose_visualisation == 'Activities with best average scores':
      st.header('Activities with best average scores in year: %d'%int(choose_year))  
      year_best_activities(int(choose_year))

    elif choose_visualisation == 'Activities with worst average scores':
      st.header('Activities with worst average scores in year: %d'%int(choose_year))  
      year_worst_activities(int(choose_year))

    elif choose_visualisation == 'Times played by month':
      st.header('Number of activities played each month in year: %d'%int(choose_year))  
      year_activities_per_month(int(choose_year))
      st.write('The summer months (July and August) and winter months (December and January) are usually the months with the least played activities')
      
    elif choose_visualisation == 'Times played by day of week':
      st.header('Number of activities played by day of week in year: %d'%int(choose_year))  
      year_activities_day_of_week(int(choose_year))
      st.write('Activities are usually played less on the weekend')
    else:
      'We are working on this'  


else:  
  choose_activity = st.sidebar.selectbox(
    'Choose activity ID',
    ['Activity ID', '1', '267', '268','275','276','277','278','279','280','281',
    '282','283','284','285','286','287','288','289','290','291','292','293','294',
    '295','296','297','310','346','350','351','352','353', '354', '355','356','357',
    '358','359','360','361','362','363','379','380', '381', '382', '383', '384','385',
    '388','389','390','391','392','393','473','474','475','476','477','478','480',
    '481','482','483','484','','485','487','488','492','494','497','577','578','579','580',
    '581','582','583','584','585','587','626','634','635','669','716','717','718','719',
    '720','721','722','723','724','725','726','728','729','735','736','737','738','739','740',
    '741','742','743','744','746','747','748','749','750','751','752','753','754','755','756','757',
    '758','759','760','761','762','763','764','765','767','768','769','771','772','773','775','776','777','778',
    '779','781','782','783','784','785','786','787','791','793','794','795','798','799',
    '800','801','802','803','804','805','806','807','808','809','810','811','812','814','815','816','817',
    '818','819','820','821','822','824','825','827','828','829','830','833','834','836','837','838','839','840',
    '841','842','845','846','847','848','850','851','855','857','866','867','869','872','875','878','881','885','887','890','900','905',
    '909','930','935','938','954','957','959','960','963','968','975','978','989','993',
    '1001','1004','1007','1008','1016','1019','1020','1022','1025','1028','1029','1035','1040','1046','1049','1053',
    '1055','1058','1059','1061','1064','1065','1067','1070','1071','1077','1085','1086','1092','1094','1097',
    '1100','1103','1106','1109','1112','1113','1115','1116','1119','1121','1122','1124','1127','1130','1131','1137','1148','1154','1155','1157','1160',
    '1163','1170','1174','1177','1179','1180','1183','1186','1191','1194','1198','1201','1203','1206',
    '1207', '1210','1213','1215','1216','1221','1224','1230','1241','1247','1248','1256','1257','1259','1262',
    '1210','1263','1277','1283','1284','1294','1296','1302','1308','1314','1344','1350','1356','1362','1368','1374','1380',
    '1383','1389','1398','1402','1406','1417','1421','1423','1427','1431','1435','1438','1441','1444','1447','1450','1453',
    '1456','1459','1462','1465','1467','1470','1473','1475','1478','1481','1482','1483','1488','1491','1497'])

  if choose_activity == 'Activity ID':
      st.write('Please choose an activity ID')
  else:
    choose_visual = st.sidebar.selectbox(
    'Choose a visualisation of activity: ' + choose_activity,
    ['Choose visualisation', 'activity type', 'times played', 'times played by year',
    'times played by month', 'times played by day of week', 'activity result variables', 'activity domains',
    'difficulty levels', 'mean score per difficulty level', 'mean time per difficulty level', 'score distributions',
    'specific patient score', 'activity times', 'activity realizacion', 'activity intentos', 'success v errors',
    'game activity sublevels'])
    if choose_visual == 'Choose visualisation':
      'Please choose a visualisation option'
    elif choose_visual == 'activity type':
      st.header('Activity %d type'%int(choose_activity))  
      activity_types(int(choose_activity))
    elif choose_visual == 'times played':
      st.header('Distribution showing number of times activity %d was played'%int(choose_activity))  
      times_played(int(choose_activity))
      st.write('The most common amount of times a user plays an activity is once, however some users do play activities over 100 times.')
    elif choose_visual == 'times played by year':
      st.header('number of times activity %d was played per year'%int(choose_activity))
      times_played_year(int(choose_activity))
      st.write('Note that 2022 is not a full year of data, only going up to July 2022')  
    elif choose_visual == 'times played by month':
      st.header('number of times %d activity was played per month'%int(choose_activity)) 
      times_played_month(int(choose_activity))
      st.write('Activities are usually played less in the winter and summer months') 
    elif choose_visual == 'times played by day of week':
      st.header('number of times activity %d was played by day of week'%int(choose_activity))  
      times_played_dayofweek(int(choose_activity))
      st.write('Activities are usually played less on the weekend') 
    elif choose_visual == 'activity result variables':
      st.header('Result variables of activity %d'%int(choose_activity))  
      show_result_variable(int(choose_activity))  
    elif choose_visual == 'activity domains':
      st.header('Domains worked on in activity %d'%int(choose_activity))  
      activity_domains(int(choose_activity))    
    elif choose_visual == 'difficulty levels':
      st.header('Difficulty levels of activity %d'%int(choose_activity))  
      difficulty_levels(int(choose_activity))    
    elif choose_visual == 'mean score per difficulty level':
      st.header('different difficulty levels mean scores on activity %d'%int(choose_activity))  
      difficulty_scores(int(choose_activity))  
    elif choose_visual == 'mean time per difficulty level':
      st.header('different difficulty levels mean times on activity %d'%int(choose_activity))  
      difficulty_times(int(choose_activity))    
    elif choose_visual == 'score distributions':
      st.header('Activity %d score distributions'%int(choose_activity))  
      scores(int(choose_activity))  
    elif choose_visual == 'specific patient score':
      enter_patient = st.sidebar.text_input("Enter patient ID e.g., for activity ID:1 : patient ID:4", key="patient_id")
      try:
        st.header('Patient score compared to patient population')
        score_activity_patient(int(choose_activity), int(enter_patient))
      except Exception as e:
        st.write('Enter a valid patient ID for activity %d'%int(choose_activity))
        print(e)  
    
    elif choose_visual == 'activity times':
      st.header('Average time taken to complete activity %d'%int(choose_activity))  
      activity_time(int(choose_activity))
    elif choose_visual == 'activity realizacion':
      st.header('Activity %d performance types'%int(choose_activity))  
      realizacion(int(choose_activity)) 
    elif choose_visual == 'activity intentos':
      st.header('Average attempts on activity %d per difficulty level'%int(choose_activity))  
      intentos(int(choose_activity))   
    elif choose_visual == 'success v errors':
      st.header('Proportion of hits compared to errors on activity %d'%int(choose_activity))  
      aciertos_errores(int(choose_activity))  
    else:
      st.header('Variation in mean and difficulty for each activity %d sublevel'%int(choose_activity))  
      game_fase_pantalla_subenivel(int(choose_activity)) 




