import pandas as pd
import os



folder_path = '/Users/faizahmad/Desktop/00 Shisir/LifeEx Data and CSV files/Life Expectency Data'

# List all the csv files in the folder

csv_files = [x for x in os.listdir(folder_path) if x.endswith('.CSV')] ## this is an example of list comprehension
csv_files

# lets create an empty list to store the dataframes
dataframes = []

# lets loop over the list of CSV files and read each one

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

print(type(dataframes))
print(len(dataframes)) ## A total of 20 dataframes, stored as a list


# Modify the processing to include FIPS code with 5-digit padding

reduced_dataframe = []

for i in range(0, 20):
    
    ## lets extract the total life expectency and only of the age group less than 1 year olds.
    df2 = dataframes[i].loc[(dataframes[i]['race_name'] == 'Total') & (dataframes[i]['age_name'] == '<1 year')]

    ## lets remove empty cells
    df3 = df2.dropna()

    ## As the dataframe consists of life expectencey at the state level as well
    ## lets gather only those with county, since fips for state end at 56, will set the condition to be greater than this to get the data at the county level.
    df4 = df3.loc[(df3['fips'] > 60)]

    ## lets delete these columns.
    df5 = df4.drop(['measure_id', 'location_id', 'measure_name', 'race_id', 'race_name', 'sex_id', 'sex_name', 'age_group_id',
     'age_name', 'metric_id', 'metric_name', 'upper', 'lower'], axis=1)

    ## Convert FIPS to integer then format with 5-digit padding
    df5['fips'] = df5['fips'].astype(int).astype(str).str.zfill(5)

    ## lets rename the columns
    df5 = df5.rename(columns={'val': 'MeanLifeExpectency', 'fips': 'fips'})
    
    ## Reorder columns: location_name, fips, year, MeanLifeExpectency
    df5 = df5[['location_name', 'fips', 'year', 'MeanLifeExpectency']]

    reduced_dataframe.append(df5)


final_df=pd.concat(reduced_dataframe,ignore_index=True)
final_df
final_df.to_csv('/Users/faizahmad/Desktop/LE Crop data/LE_Crop_data.csv', index=False)




# List all the csv files in the folder

csv_files = [x for x in os.listdir(folder_path) if x.endswith('.CSV')] ## this is an example of list comprehension
csv_files


reduced_dataframe=[]

for i in range (0,20):
    
    ## lets extract the total life expectency and only of the age group less than 1 year olds.

    df2=dataframes[i].loc[(dataframes[i]['race_name'] == 'Total') & (dataframes[i]['age_name'] == '<1 year')]

    ## lets remove empty cells

    df3=df2.dropna()

    ## As the dataframe consists of life expectencey at the state level as well
    ## lets gather only those with county, since fips for state end at 56, will set the condition to be greater than this to get the data at the county level.

    df4=df3.loc[(df3['fips'] > 60)]


    ## lets delete these columns. 
    df5 = df4.drop(['measure_id', 'location_id', 'fips','measure_name', 'race_id', 'race_name', 'sex_id', 'sex_name', 'age_group_id',
     'age_name', 'metric_id', 'metric_name', 'upper', 'lower'], axis=1)

    ## lets rename the column val to mean life expectency. 
    df5 = df5.rename(columns={'val': 'MeanLifeExpectency'})

    reduced_dataframe.append(df5)


final_df=pd.concat(reduced_dataframe,ignore_index=True)
final_df