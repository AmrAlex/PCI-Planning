import pandas as pd
import numpy as np
import os
import math
import random

def get_inputs_within_ranges(num_inputs, ranges,msgs,function,error):
    inputs = []
    for i in range(num_inputs):
        while True:
            user_input = input(msgs[i])
            try:
                user_input = int(user_input)
                if user_input < ranges[i][0] or user_input > ranges[i][1]:
                    raise ValueError
            except ValueError:
                print(error[i])
                continue


            # If input is within range, append to list of inputs and break the loop
            inputs.append(user_input)
            break
    function(inputs)
# Example usage: Get 3 inputs from user within multiple ranges

def plan_one_site(s):
    PCI_start=s[2]
    PCI_End=s[3]
    preamble=s[4]
    site_name=input("enter valid site name ")
    df_concat_site = pd.read_csv("PCI_Output_detailed.csv")
    df_concat_site.loc[df_concat_site['site_name'] == int(site_name), 'site_name_SSS'] = None
    df_concat_site.loc[df_concat_site['site_name'] == int(site_name), 'site_name_RSI_site'] = None
    df_concat_site.loc[df_concat_site['nbr_site_name'] == int(site_name), 'nbr_SSS'] = None
    df_concat_site.loc[df_concat_site['nbr_site_name'] == int(site_name), 'nbr_site_name_RSI'] = None

    if int(site_name) in df_concat_site['site_name'].values:
     for index,row in df_concat_site.iterrows():
        if pd.isnull(row['site_name_SSS']):
            if int(preamble)==1:
                random_number_pre = random.randint(0, 135/3)
            else:
                random_number_pre = random.randint(0, 837/3)
            random_number = random.randint(PCI_start, PCI_End)
            df_filter_site=df_concat_site.loc[df_concat_site['site_name']== row['site_name']]
            if random_number not in df_filter_site['nbr_SSS']:
                    df_concat_site.loc[df_concat_site['site_name'] == row['site_name'], 'site_name_SSS'] = random_number
                    df_concat_site.loc[df_concat_site['nbr_site_name'] == row['site_name'], 'nbr_SSS'] = random_number
                                   
            if random_number_pre not in df_filter_site['nbr_site_name_RSI']:
                    df_concat_site.loc[df_concat_site['site_name'] == row['site_name'], 'site_name_RSI_site'] = random_number_pre
                    df_concat_site.loc[df_concat_site['nbr_site_name'] == row['site_name'], 'nbr_site_name_RSI'] = random_number_pre

     #df_concat_site.to_csv('PCI_Output_detailed_one_site.csv', index=False)
     df_PCI_only_one = pd.DataFrame()
     df_PCI_only_one['site_name']=df_concat_site['site_name']
     df_PCI_only_one['cell name']=df_concat_site['Cell_Name']
     df_PCI_only_one['SSS_cell']=df_concat_site['site_name_SSS']
     df_PCI_only_one['PSS_cell']=df_concat_site['PSS_cell']
     df_PCI_only_one['PCI_cell']=df_concat_site['PSS_cell']+(df_concat_site['site_name_SSS'])*3
     df_PCI_only_one['RSI']=df_concat_site['PSS_cell']+(df_concat_site['site_name_RSI_site'])*3
     df_PCI_only_one = df_PCI_only_one.drop_duplicates()
     df_PCI_only_one_filter = df_PCI_only_one[df_PCI_only_one['site_name'] == int(site_name)]
     df_PCI_only_one_filter.to_csv('PCI_Output_one_site.csv', index=False)
     print('done Bye Bye ')

pd.options.mode.chained_assignment = None 


def nbr_plan():
 global db_sites,l18_nbrs,l21_nbrs
 db_sites = pd.read_csv("DB.csv")
 cells = db_sites[['SITE', 'Cell Name', 'LTE_Band',
                'Latitude', 'Longitude', 'Azimuth']]
 cells_18 = cells.query("LTE_Band == 'L18'")
 cells_18.reset_index(inplace=True, drop=True)
 cells_21 = cells.query("LTE_Band == 'L21'")
 cells_21.reset_index(inplace=True, drop=True)
 cells_18_copy = cells_18.copy()
 cells_21_copy = cells_21.copy()
 l18_nbrs = {'Cell_Name': [], 'Nbrs': []}
 l21_nbrs = {'Cell_Name': [], 'Nbrs': []}
 
 distance = 0

# get user input for city type.
# Using a while loop to handle invalid inputs
 while True:
    print("""What is the terrain type ?
1. Dense-Urban (500 m)
2. Urban (800 m)
3. Sub-Urban (1000 m)
4. Rural (2 km) """)

    city_type = input("What is your choice no.: ")
    try:
        if int(city_type) not in range(1, 5):
            print(
                "Invalid input, please choose one of the provided options")
            continue
        else:
            if int(city_type) == 1:
                distance = 500
            elif int(city_type) == 2:
                distance = 800
            elif int(city_type) == 3:
                distance = 1000
            elif int(city_type) == 4:
                distance = 2000
                
            break

    except:
        print("Invalid input, please choose one of the provided options\n")

# iterate over cells of L18 to get nearest site to each cell and angle
 for index, cell_phy_data in cells_18.iterrows():
    cells_18_copy["Latitude1"] = cell_phy_data.Latitude
    cells_18_copy["Longitude1"] = cell_phy_data.Longitude
    curr_cellname = cell_phy_data['Cell Name']

    az = cell_phy_data.Azimuth
    if az >= 360:
        az = az % 360

    cells_18_copy["Azimuth1"] = az

    # Calculate distance
    cells_18_copy["RLat2"] = np.radians(
        cells_18_copy["Latitude"].astype(float))
    cells_18_copy["RLon2"] = np.radians(
        cells_18_copy["Longitude"].astype(float))
    cells_18_copy["RLat1"] = np.radians(
        cells_18_copy["Latitude1"].astype(float))
    cells_18_copy["RLon1"] = np.radians(
        cells_18_copy["Longitude1"].astype(float))
    cells_18_copy["dlon"] = cells_18_copy["RLon2"] - cells_18_copy["RLon1"]
    cells_18_copy["dlat"] = cells_18_copy["RLat2"] - cells_18_copy["RLat1"]
    cells_18_copy["a"] = \
        np.sin(cells_18_copy["dlat"]/2)**2 \
        + np.cos(cells_18_copy["RLat1"]) * np.cos(cells_18_copy["RLat2"]) \
        * np.sin(cells_18_copy["dlon"]/2)**2
    cells_18_copy["c"] = 2 * np.arcsin(np.sqrt(cells_18_copy["a"]))
    cells_18_copy["distance"] = cells_18_copy["c"] * 6371 * 1000

    # Calculate the angel between the site and other sites
    cells_18_copy["y"] = np.sin(
        cells_18_copy["dlon"]) * np.cos(cells_18_copy["RLat2"])
    cells_18_copy["x"] = np.cos(cells_18_copy["RLat1"]) * np.sin(cells_18_copy["RLat2"]) - np.sin(cells_18_copy["RLat1"]) * \
        np.cos(cells_18_copy["RLat2"]) * np.cos(cells_18_copy["dlon"])
    cells_18_copy["bearing"] = np.degrees(
        np.arctan2(cells_18_copy["y"], cells_18_copy["x"]))
    cells_18_copy["angle"] = (cells_18_copy["bearing"] + 360) % 360

    # Sort by distance
    cells_18_copy.sort_values(by="distance", inplace=True)

    # Check whether distance and angle criteria are satisfied or not
    dis_chk = cells_18_copy["distance"].astype(float) < float(distance)

    if az >= 0 and az < 90:
        low_bound = (az - 90) % 360
        high_bound = (az + 90) % 360
        ang_chk = (cells_18_copy["angle"] <= high_bound) | (
            cells_18_copy["angle"] >= low_bound)

    elif az >= 90 and az < 270:
        low_bound = (az - 90) % 360
        high_bound = (az + 90) % 360
        ang_chk = (cells_18_copy["angle"] <= high_bound) & (
            cells_18_copy["angle"] >= low_bound)

    elif az >= 270 and az < 360:
        low_bound = (az - 90) % 360
        high_bound = (az + 90) % 360
        ang_chk = (cells_18_copy["angle"] <= high_bound) | (
            cells_18_copy["angle"] >= low_bound)

    cells_18_copy.loc[(dis_chk & ang_chk) | (
        cells_18_copy['SITE'] == cell_phy_data.SITE), "Nbr"] = "yes"

    cells_18_copy.loc[~((dis_chk & ang_chk) | (
        cells_18_copy['SITE'] == cell_phy_data.SITE)), "Nbr"] = "no"

    # Fill the dictionary with the nbrs of each cell
    nbrs_series = cells_18_copy[cells_18_copy['Nbr'] == 'yes']['Cell Name']

    # cells_18_copy["Nbr"] = np.nan

    for nbr in nbrs_series:
        if nbr == curr_cellname:
            continue
        else:
            l18_nbrs['Cell_Name'].append(curr_cellname)
            l18_nbrs['Nbrs'].append(nbr)

# iterate over cells of L21 to get nearest site to each cell and angle
 for index, cell_phy_data in cells_21.iterrows():
    cells_21_copy["Latitude1"] = cell_phy_data.Latitude
    cells_21_copy["Longitude1"] = cell_phy_data.Longitude
    curr_cellname = cell_phy_data['Cell Name']

    # l21_nbrs[curr_cellname] = []

    az = cell_phy_data.Azimuth
    if az >= 360:
        az = az % 360

    cells_21_copy["Azimuth1"] = az

    # Calculate distance
    cells_21_copy["RLat2"] = np.radians(
        cells_21_copy["Latitude"].astype(float))
    cells_21_copy["RLon2"] = np.radians(
        cells_21_copy["Longitude"].astype(float))
    cells_21_copy["RLat1"] = np.radians(
        cells_21_copy["Latitude1"].astype(float))
    cells_21_copy["RLon1"] = np.radians(
        cells_21_copy["Longitude1"].astype(float))
    cells_21_copy["dlon"] = cells_21_copy["RLon2"] - cells_21_copy["RLon1"]
    cells_21_copy["dlat"] = cells_21_copy["RLat2"] - cells_21_copy["RLat1"]
    cells_21_copy["a"] = \
        np.sin(cells_21_copy["dlat"]/2)**2 \
        + np.cos(cells_21_copy["RLat1"]) * np.cos(cells_21_copy["RLat2"]) \
        * np.sin(cells_21_copy["dlon"]/2)**2
    cells_21_copy["c"] = 2 * np.arcsin(np.sqrt(cells_21_copy["a"]))
    cells_21_copy["distance"] = cells_21_copy["c"] * 6371 * 1000

    # Calculate the angel between the site and other sites
    cells_21_copy["y"] = np.sin(
        cells_21_copy["dlon"]) * np.cos(cells_21_copy["RLat2"])
    cells_21_copy["x"] = np.cos(cells_21_copy["RLat1"]) * np.sin(cells_21_copy["RLat2"]) - np.sin(cells_21_copy["RLat1"]) * \
        np.cos(cells_21_copy["RLat2"]) * np.cos(cells_21_copy["dlon"])
    cells_21_copy["bearing"] = np.degrees(
        np.arctan2(cells_21_copy["y"], cells_21_copy["x"]))
    cells_21_copy["angle"] = (cells_21_copy["bearing"] + 360) % 360

    # Sort by distance
    cells_21_copy.sort_values(by="distance", inplace=True)

    # Check whether distance and angle criteria are satisfied or not
    dis_chk = cells_21_copy["distance"].astype(float) < float(distance)

    if az >= 0 and az < 90:
        low_bound = (az - 90) % 360
        high_bound = (az + 90) % 360
        ang_chk = (cells_21_copy["angle"] <= high_bound) | (
            cells_21_copy["angle"] >= low_bound)

    elif az >= 90 and az < 270:
        low_bound = (az - 90) % 360
        high_bound = (az + 90) % 360
        ang_chk = (cells_21_copy["angle"] <= high_bound) & (
            cells_21_copy["angle"] >= low_bound)

    elif az >= 270 and az < 360:
        low_bound = (az - 90) % 360
        high_bound = (az + 90) % 360
        ang_chk = (cells_21_copy["angle"] <= high_bound) | (
            cells_21_copy["angle"] >= low_bound)

    cells_21_copy.loc[(dis_chk & ang_chk) | (
        cells_21_copy['SITE'] == cell_phy_data.SITE), "Nbr"] = "yes"

    # Fill the dictionary with the nbrs of each cell
    nbrs_series = cells_21_copy[cells_21_copy['Nbr'] == 'yes']['Cell Name']

    cells_21_copy["Nbr"] = np.nan

    for nbr in nbrs_series:
        if nbr == curr_cellname:
            continue
        else:
            # l21_nbrs[curr_cellname].append(nbr)
            l21_nbrs['Cell_Name'].append(curr_cellname)
            l21_nbrs['Nbrs'].append(nbr)

 df_l18_nbrs = pd.DataFrame(l18_nbrs)
 df_l21_nbrs = pd.DataFrame(l21_nbrs)
# Create the list where we 'll capture the cells that appear for 1st time,
# add the 1st row and we start checking from 2nd row until end of df
 l18_startCells = [1]
 for row in range(2, len(df_l18_nbrs)+1):
    if (df_l18_nbrs.loc[row-1, 'Cell_Name'] != df_l18_nbrs.loc[row-2, 'Cell_Name']):
        l18_startCells.append(row)

 l21_startCells = [1]
 for row in range(2, len(df_l21_nbrs)+1):
    if (df_l21_nbrs.loc[row-1, 'Cell_Name'] != df_l21_nbrs.loc[row-2, 'Cell_Name']):
        l21_startCells.append(row)

 with pd.ExcelWriter('nbrs.xlsx', engine='xlsxwriter') as writer:
    df_l18_nbrs.to_excel(writer, sheet_name='L18_Nbrs', index=False)
    df_l21_nbrs.to_excel(writer, sheet_name='L21_Nbrs', index=False)
    cells_18_copy.to_excel(writer, sheet_name='L18_Raw_Data', index=False)
    cells_21_copy.to_excel(writer, sheet_name='L21_Raw_Data', index=False)

    workbook = writer.book
    worksheet18 = writer.sheets['L18_Nbrs']
    worksheet21 = writer.sheets['L21_Nbrs']
    merge_format = workbook.add_format(
        {'align': 'center', 'valign': 'vcenter', 'border': 1})

    lastRow18 = len(df_l18_nbrs)
    lastRow21 = len(df_l21_nbrs)

    for row in l18_startCells:
        try:
            endRow = l18_startCells[l18_startCells.index(row)+1]-1
            if row == endRow:
                worksheet18.write(
                    row, 0, df_l18_nbrs.loc[row-1, 'Cell_Name'], merge_format)
            else:
                worksheet18.merge_range(
                    row, 0, endRow, 0, df_l18_nbrs.loc[row-1, 'Cell_Name'], merge_format)
        except IndexError:
            if row == lastRow18:
                worksheet18.write(
                    row, 0, df_l18_nbrs.loc[row-1, 'Cell_Name'], merge_format)
            else:
                worksheet18.merge_range(
                    row, 0, lastRow18, 0, df_l18_nbrs.loc[row-1, 'Cell_Name'], merge_format)

    for row in l21_startCells:
        try:
            endRow = l21_startCells[l21_startCells.index(row)+1]-1
            if row == endRow:
                worksheet21.write(
                    row, 0, df_l21_nbrs.loc[row-1, 'Cell_Name'], merge_format)
            else:
                worksheet21.merge_range(
                    row, 0, endRow, 0, df_l21_nbrs.loc[row-1, 'Cell_Name'], merge_format)
        except IndexError:
            if row == lastRow21:
                worksheet21.write(
                    row, 0, df_l21_nbrs.loc[row-1, 'Cell_Name'], merge_format)
            else:
                worksheet21.merge_range(
                    row, 0, lastRow21, 0, df_l21_nbrs.loc[row-1, 'Cell_Name'], merge_format)
                
nbr_plan()

############## PCI planning start 
import random
def RSI_PCI_plan(s):
 technolgy=s[0]
 planning_type=s[1]
 PCI_start=s[2]
 PCI_End=s[3]
 preamble=s[4]
 cells = db_sites[['Cell Name', 'LTE_Band','SITE','Azimuth','Latitude','Longitude']]
 cells['site_band'] = cells['SITE']+cells['LTE_Band']
 cells['sector_number'] = None
 cells = cells.rename(columns={'Cell Name': 'Cell_Name'})

 def label_records(group):
    group['sector_number'] = group['Azimuth'].rank(method='dense')
    return group

# group the DataFrame by the 'Name' column and apply the function to each group
 df_labeled = cells.groupby('site_band',group_keys=False).apply(label_records)
# get user input for city type.
# Using a while loop to handle invalid inputs

 if int(planning_type)==2:
            df_l18_nbrs = pd.DataFrame(l18_nbrs)
            df_l18_nbrs['band'] = 18
            df_l21_nbrs = pd.DataFrame(l21_nbrs)
            df_l21_nbrs['band'] = 21
            df_concat = pd.concat([df_l18_nbrs, df_l21_nbrs])
            df_concat['site_name'] = df_concat['Cell_Name'].str.slice(stop=-1)
            df_concat['nbr_site_name'] = df_concat['Nbrs'].str.slice(stop=-1)
            df_concat['site_name_SSS'] = None
            df_concat['nbr_SSS'] = None

            for index, row in df_concat.iterrows():
                if row['site_name_SSS']==None:
                    while True:
                        if int(technolgy)==1:
                          random_number = random.randint(PCI_start, PCI_End)
                        else:
                          random_number = random.randint(PCI_start, PCI_End)
                        df_filter=df_concat.loc[df_concat['site_name']== row['site_name']]
                        if random_number not in df_filter['nbr_SSS']:
                            df_concat.loc[df_concat['site_name'] == row['site_name'], 'site_name_SSS'] = random_number
                            df_concat.loc[df_concat['nbr_site_name'] == row['site_name'], 'nbr_SSS'] = random_number
                            break
            df_concat= pd.merge(df_concat, df_labeled[['Cell_Name','sector_number']], on='Cell_Name', how='left')

            def assign_value(x):
                    if int(x)==1:
                        return 0
                    elif int(x)==2:
                        return 1
                    elif int(x)==3:
                        return 2
                    else:
                        return None
                # Apply the function to create the "NewCol" column
            df_concat["PSS_cell"] = df_concat['sector_number'].apply(assign_value)
            # Print the updated DataFrame
            #df_concat.to_csv('PCI_Output_detailed.csv', index=False)
            df_PCI_only = pd.DataFrame()

            df_PCI_only['cell name']=df_concat['Cell_Name']
            df_PCI_only['SSS_cell']=df_concat['site_name_SSS']
            df_PCI_only['PSS_cell']=df_concat['PSS_cell']
            df_PCI_only['PCI_cell']=df_concat['PSS_cell']+(df_concat['site_name_SSS'])*3
            df_PCI_only = df_PCI_only.drop_duplicates()

            #df_PCI_only.to_csv('PCI_Output.csv', index=False)

            df_concat['site_name_RSI_site'] = None
            df_concat['nbr_site_name_RSI'] = None

            for index,row in df_concat.iterrows():
                            if pd.isnull(row['site_name_RSI_site']):
                                if int(preamble)==1:
                                   random_number_pre = random.randint(0, 135/3)
                                else:
                                   random_number_pre = random.randint(0, 837/3)
                            df_filter=df_concat.loc[df_concat['site_name']== row['site_name']]
      
                            if random_number_pre not in df_filter['nbr_site_name_RSI']:
                               df_concat.loc[df_concat['site_name'] == row['site_name'], 'site_name_RSI_site'] = random_number_pre
                               df_concat.loc[df_concat['nbr_site_name'] == row['site_name'], 'nbr_site_name_RSI'] = random_number_pre

            df_concat.to_csv('PCI_Output_detailed.csv', index=False)

            df_PCI_only = pd.DataFrame()

            df_PCI_only['Cell Name']=df_concat['Cell_Name']
            df_PCI_only['SSS_cell']=df_concat['site_name_SSS']
            df_PCI_only['PSS_cell']=df_concat['PSS_cell']
            df_PCI_only['PCI_cell']=df_concat['PSS_cell']+(df_concat['site_name_SSS'])*3
            df_PCI_only['RSI']=df_concat['PSS_cell']+(df_concat['site_name_RSI_site'])*3
            df_PCI_only = df_PCI_only.drop_duplicates()
            merged_df = pd.merge(df_PCI_only, db_sites[['Cell Name','Latitude','Longitude']], on='Cell Name')
            merged_df.to_csv('PCI_Output.csv', index=False)
            print('done Bye Bye ')

            

 elif int(planning_type)==1:
            plan_one_site(s)


get_inputs_within_ranges(5, [(1,2), (1,2), (0,335),(0,335),(1,2)],["""Which technolgy do you want to plan ?
1. LTE
2. 5G
 ""","""DO u want to plan new site or the whole input ?
1. new site
2. whole_input
 ""","Enter PCI group start :","Enter PCI group End: ","""Which RSI format you want to use ?
            1. short preamble 
            2. long preamble
            """],RSI_PCI_plan,['enter a valid value 1 or 2','enter a valid value 1 or 2','enter a valid PCI start value from 0 to 167 for LTE and from 0 to 355 for 5G','enter a valid PCI start value from 0 to 167 for LTE and from 0 to 355 for 5G','enter a valid value 1 or 2'])


