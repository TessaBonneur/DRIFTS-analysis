import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from jcamp import jcamp_readfile
from typing import Literal

'''
This script is used to analyze DRIFTS spectra from either DeNOx or LP IR VMB setups. It defines the following functions:
- read_logfile: Reads the logfile from the specified path and formats it.
- parse_spectra: Reads the spectra from the specified path and formats them.
- quick_plot: Makes a quick plot of the raw spectra and saves it as a png file.
- merge_spectra_logfile: Merges the logfile and spectra based on the time of the spectra. It also makes a column for the number of the spectrum.
- background_correct_by_temperature: Performs background correction by subtracting a 'background' spectrum with the closest temperature from the 'reaction' spectrum.
- ...and many other functions used for data analysis which might not be useful for everyone.
'''

# define the columns for a particular logfile
def read_logfile(setup: Literal['LP IR VMB', 'DeNOx'] = 'DeNOx', logfile_path: str = 'data', delete_previous: bool = False):
    '''
    Read the logfile from the specified path and format it.
    
    Parameters:
    - setup: Type of setup used for the experiment. Options are 'LP IR VMB' or 'DeNOx'.
    - logfile_path: Path to the folder containing the logfile.
    - delete_previous: If True, deletes any previously parsed logfile.csv in the folder before reading.

    Returns:
    - logfile: DataFrame with the formatted logfile data.
    '''

    if delete_previous == True:
        # delete the logfile.csv file if it exists
        try:
            os.remove(logfile_path + '/logfile.csv')
            print('Deleted previous logfile.csv file.')
        except:
            print('No previous logfile.csv file found.')

    # Try to load previously parsed logfile
    try:
        logfile = pd.read_csv(logfile_path + '/logfile.csv', index_col=0)
        print('Logfile loaded successfully. If reloading is required, delete the logfile.csv file in the folder.')
        return logfile
    except:
        print('No logfile found. Loading logfile from the specified path...')

        # Make a list of the txt files in the folder
        logfile_files = sorted([file for file in os.listdir(logfile_path) if file.endswith('.txt')])

        # Check if the logfile files are empty
        if len(logfile_files) == 0:
            raise ValueError('No logfile found in the specified path.')

        # if there are multiple logfiles, combine them, else: just read the one
        if len(logfile_files) > 1:
            print('Multiple logfiles found! Combining them...')
            logfile = pd.concat([pd.read_csv(logfile_path + '\\' + i, sep='\t', skiprows=2) for i in logfile_files])
        else:
            logfile = pd.read_csv(logfile_path + '\\' + logfile_files[0], sep='\t', skiprows=2)  # Adjust `sep` and `skiprows` as needed

        print('Converting logfile date and time to pandas DateTime...')
        try:
            if setup == 'DeNOx':
                logfile['DateTime'] = pd.to_datetime(logfile['Date'] + ' ' + logfile['Time'], format='mixed', dayfirst=True, errors='coerce')

            if setup == 'LP IR VMB':
                logfile['DateTime'] = pd.to_datetime(logfile['Date'] + ' ' + logfile['Time'], format='%m/%d/%Y %H:%M:%S')

            global reaction_date
            reaction_date = str(logfile['DateTime'].iloc[0].date())  # Store the date of the first measurement for later use

        except Exception as e:
            raise Exception(f'Error with datetime conversion: {e}')

        # Rename columns based on the setup
        if setup == 'LP IR VMB':
            logfile.rename(columns={
                'SP N2-bub': 'N2_bubbler_sp',
                'Flow N2-bub': 'N2_bubbler_flow',
                'SP N2': 'N2_sp',
                'flow N2': 'N2_flow',
                'Sp CO2': 'CO2_sp',
                'flow CO2': 'CO2_flow',
                'SP H2': 'H2_sp',
                'flow H2': 'H2_flow',
                'SP CO': 'CO_sp',
                'flow CO': 'CO_flow',
                'SP O2': 'O2_sp',
                'flow O2': 'O2_flow',
                'Oven actual SP': 'Oven_actual_sp',
                'Oven temp': 'Oven_temp',
                'oven ramp': 'Oven_ramp'
            }, inplace=True)

            # then rename O2 to CO
            logfile.rename(columns={
                'O2_sp': 'CO_sp',
                'O2_flow': 'CO_flow'
            }, inplace=True)

        if setup == 'DeNOx':
            logfile.rename(columns={
                'SP He Low': 'He_low_sp',
                'Flow He Low': 'He_low_flow',
                'SP 2%CO': 'CO_sp',
                'Flow 2%CO': 'CO_flow',
                'SP H2': 'H2_sp',
                'Flow H2': 'H2_flow',
                'SP 1%NO': 'NO_sp',
                'Flow 1%NO': 'NO_flow',
                'SP 0.5%Propene': 'Propene_sp',
                'Flow 0.5%Propene': 'Propene_flow',
                'sp O2': 'O2_sp',
                'Flow O2': 'O2_flow',
                'Oven actual SP': 'Oven_actual_sp',
                'Oven Temp': 'Oven_temp',
                'Oven Ramp': 'Oven_ramp',
                'Target Oven SP': 'Target_Oven_sp',
                'Oven Temp internal': 'Oven_temp_internal',
                'Oven %': 'Oven_percent'
            }, inplace=True)

        print('Logfile read successfully.\n')

        # Export the logfile to a csv file 
        logfile.to_csv(logfile_path + '/logfile.csv', index=False)


    # Preview the logfile
    return logfile

# Define variables with the names of the columns in the logfile. Callable with DRIFTS_package.Date, DRIFTS_package.Time, etc.
Date = 'Date'
Time = 'Time'
DateTime = 'DateTime'
N2_bubbler_sp = 'N2_bubbler_sp'
N2_bubbler_flow = 'N2_bubbler_flow'
N2_sp = 'N2_sp'
N2_flow = 'N2_flow'
CO2_sp = 'CO2_sp'
CO2_flow = 'CO2_flow'
H2_sp = 'H2_sp'
H2_flow = 'H2_flow'
CO_sp = 'CO_sp'
CO_flow = 'CO_flow'
O2_sp = 'O2_sp'
O2_flow = 'O2_flow'
Oven_actual_sp = 'Oven_actual_sp'
Oven_temp = 'Oven_temp'
Oven_ramp = 'Oven_ramp'
Target_Oven_sp = 'Target_Oven_sp'
Oven_temp_internal = 'Oven_temp_internal'
Oven_percent = 'Oven_percent'
He_low_sp = 'He_low_sp'
He_low_flow = 'He_low_flow'
CO_sp = 'CO_sp'
CO_flow = 'CO_flow'
NO_sp = 'NO_sp'
NO_flow = 'NO_flow'
Propene_sp = 'Propene_sp'
Propene_flow = 'Propene_flow'


def xaxis_inversion(spectra: pd.DataFrame):
    """
    Invert the x-axis of the spectra DataFrame.

    Parameters:
    - spectra: DataFrame with the spectra to invert. The index should be the filenames and the columns should be the wavenumbers.

    Returns:
    - Inverted spectra DataFrame.
    """
    # Invert the x-axis (wavenumber) for better visualization
    print('Inverting x-axis. If this is not desired, set invert_xaxis = False')
    return spectra.iloc[:, ::-1]  # Reverse the order of columns


def parse_spectra(spectra_path: str = 'data', invert_xaxis = True, delete_previous = False):
    '''
    Parse the spectra from the specified path. If the spectra have been parsed before, it will load them from the csv file.

    Parameters:
    - spectra_path: Path to the folder containing the spectra files.
    - invert_xaxis: Boolean to invert the x-axis (wavenumber) for better visualization. Default is True.
    - delete_previous: Boolean to delete the previous raw_spectra.csv file if it exists. Default is False.

    Returns:
    - A DataFrame containing all spectra, with the filenames as index and the wavenumbers as columns.
    '''
    if delete_previous == True:
        # delete the raw_spectra.csv file if it exists
        try:
            os.remove(spectra_path + '/raw_spectra.csv')
            print('Deleted previous raw_spectra.csv file.')
        except:
            print('No previous raw_spectra.csv file found.')

    try:
        # this will load the spectra if they have been parsed before. Still defines a lot of variables that are used later
        print('Looking for raw_spectra.csv...')

        try:
            spectra = pd.read_csv(spectra_path + '/raw_spectra.csv', index_col=0)
        except:
            print('No csv file in data folder. Looking in the current folder...')
            spectra = pd.read_csv('raw_spectra.csv', index_col=0)

        spectra.columns = spectra.columns.astype(float)     # when importing, it reads objects instead of floats
        spectra_filenames = spectra.copy().index
        spectra_files = sorted([file for file in os.listdir(spectra_path) if file.endswith('.dx')])

        print('Found raw_spectra.csv; spectra loaded successfully. If reloading is required, specify delete_previous = True or delete the raw_spectra.csv file in the folder.')
    except:
        # If the spectra are not yet loaded, load them
        print('No csv found. Loading spectra from dx files...')

        # Read all .dx files in the folder
        spectra_files = sorted([file for file in os.listdir(spectra_path) if file.endswith('.dx')])

        if len(spectra_files) == 0:
            raise ValueError('No spectra files found in the specified path.')
        
        # Initialize an empty DataFrame to store the spectral data
        spectra = pd.DataFrame()

        # Define a custom wavenumber series to avoid rounding errors

        firstx = jcamp_readfile(os.path.join(spectra_path, spectra_files[0]))['firstx']
        lastx = jcamp_readfile(os.path.join(spectra_path, spectra_files[0]))['lastx']
        npoints = jcamp_readfile(os.path.join(spectra_path, spectra_files[0]))['npoints']
        wavenumber_list = np.linspace(firstx, lastx, npoints)

        # Read each spectrum file and append its data

        for file in spectra_files:
            spectrum = jcamp_readfile(os.path.join(spectra_path, file))
            df = pd.DataFrame((wavenumber_list, spectrum['y'])).T  # Transpose to align columns
            df.columns = ['Wavenumber', file.split('.')[0]]  # Use filename as column name
            if spectra.empty:
                spectra = df  # Initialize the DataFrame with the first file
            else:
                spectra = pd.merge(spectra, df, on='Wavenumber', how = 'outer')  # Merge on Wavenumber
        # Preview the spectral data

        spectra.index = spectra['Wavenumber']   # Puts wavenumber as index
        spectra.drop(columns=['Wavenumber'], inplace=True)
        spectra = spectra.T
        
        # invert x_axis
        try:
            if invert_xaxis == True:
                spectra = xaxis_inversion(spectra)
        except:
            print('Inverting x_axis failed.')

        # export the spectra to a csv file to avoid parsing every time!!!
        print('Exporting spectra to csv.')
        spectra.to_csv(spectra_path + '/raw_spectra.csv')

        print('Spectra parsed successfully.\n')
    return spectra

def quick_plot(spectra_to_plot: pd.DataFrame, folder, skip: int=10):  # make a quick plot of the raw spectra and save it as a png file
    '''
    Quickly plot the raw spectra and save as a PNG file.

    Parameters:
    - spectra_to_plot: DataFrame with the spectra to plot. Index should be filenames, columns are wavenumbers.
    - folder: Folder to save the plot.
    - skip: Number of spectra to skip between each plot. Default is 10.

    Returns:
    - None. Displays and saves a plot of the raw spectra.
    '''


    colormap = plt.cm.coolwarm(np.linspace(1, 0, len(spectra_to_plot)))

    title = 'Raw spectra'

    fig, ax = plt.subplots(figsize=(16, 9))
    for n in np.arange(0,len(spectra_to_plot), skip):  # plotting from 1 onwards to skip the background spectrum
        ax.plot(spectra_to_plot.columns[:], spectra_to_plot.iloc[n], c=colormap[n])
    ax.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax.set_ylabel('Intensity')
    ax.set_title(title)
    ax.set_xlim(spectra_to_plot.columns[0], spectra_to_plot.columns[-1])  # Invert x-axis
    fig.savefig(folder + '/' + title + '.png', dpi=300, bbox_inches='tight')

    plt.show()

def merge_spectra_logfile(spectra_path: str = 'data', logfile_path: str = 'data', setup: Literal['LP IR VMB', 'DeNOx'] = 'DeNOx', spectra_start_time: Literal['YYYY-MM-DD HH:MM:SS'] = None): # if spectra_start_time is None, it will use the first time in the logfile. 
    '''
    Merge spectra and logfile based on the start time of the logfile.

    Parameters:
    - spectra_path: Path to the folder containing the spectra files OR DataFrame with spectra data.
    - logfile_path: Path to the folder containing the logfile OR DataFrame with logfile data.
    - setup: Type of setup used for the experiment. Options: 'LP IR VMB' or 'DeNOx'.
    - spectra_start_time: Start time for the spectra. If None, uses the first time in the logfile.

    Returns:
    - merged_data: DataFrame with merged spectra and logfile information.
    '''

    logfile = read_logfile(setup, logfile_path)

    spectra = parse_spectra(spectra_path)

    # Make a list of the filenames in the folder
    spectra_filenames = sorted([file for file in os.listdir(spectra_path) if file.endswith('.dx')])

    # Check if the number of spectra matches the number of filenames
    if len(spectra_filenames) != len(spectra):
        raise ValueError('The number of spectra does not match the number of filenames')

    if spectra_start_time == None:
        # First time in the logfile

        print('Using first logfile time (' + str(logfile['DateTime'].iloc[0]) + ') as start time of spectra.')

        start_time = pd.to_datetime(logfile['DateTime'].iloc[0])
    
    else:
        # Your custom time
        try:
            start_time = pd.to_datetime(spectra_start_time)
        except:
            raise ValueError('The custom start_time is not in the correct format. Use YYYY-MM-DD HH:MM:SS')
        # if the date of this custom start_time is not the same as the logfile, raise an error
        if start_time.date() != logfile['DateTime'].iloc[0].date():
            raise ValueError('The date of the custom start_time does not match the date of the logfile')

    try:
        # Manually assign timestamps to spectra based on their numbering. Assumes spectra filenames are ordered (_0001, _0002, etc.)

        # Adjust this interval if needed, default is 1 spectrum per minute
        time_interval = pd.Timedelta(minutes=1)

        spectra['DateTime'] = pd.to_datetime([start_time + i * time_interval for i in range(len(spectra_filenames))])   # This assumes the first spectrum is at the start_time
        spectra.index = spectra['DateTime']
        spectra.drop(columns=['DateTime'], inplace=True)
            
        #spectra.columns = spectra.columns.astype(float)  # Convert column names to float
    except Exception as e:
        print('Error with datetime assignment: {}'.format(e))

    try:
        global merged_data

        # Convert the DateTime column in the logfile to pandas DateTime format  
        logfile['DateTime'] = pd.to_datetime(logfile['DateTime'])
        
        # Merge spectral data with logfile information
        print('Merging spectra with logfile...')

        merged_data = pd.merge_asof(spectra.sort_values('DateTime'), logfile.sort_values('DateTime'), on='DateTime')

        # Set the DateTime column as the index
        merged_data.index = merged_data['DateTime']
        merged_data.drop(columns=['DateTime'], inplace=True)

        # Make a column for the number of the spectrum
        merged_data['Number'] = [int(file.split('.')[0].split('_')[-1]) for file in spectra_filenames]

    except Exception as e:
        print('Error during merging: {}'.format(e))
    
    try:
        # Drop the spectra recorded after the end of the logfile
        merged_data = merged_data.copy()[merged_data.index <= logfile['DateTime'].iloc[-1]]
        
        # Check if it works
        print('Removed spectra after end of logfile. You can delete spectra from 0{} onwards'.format(merged_data['Number'].iloc[-1]+1))
    
    except:
        print('No spectra recorded after the end of the logfile.')


    print('Spectra and logfile merged successfully.\n')

    return merged_data


# Important! Before using background correction, make two separate dataframes where one contains the reaction spectra and the other contains the background spectra!
# This can be done by using the 'merge_spectra_logfile' function above and then filtering the dataframes based on the temperature, time, gas flows, etc.

def background_correct_by_temperature(reaction_spectra: pd.DataFrame, background_spectra: pd.DataFrame, temp_column: str = Oven_temp, return_temp_column: bool = False):
    """
    Perform background correction by subtracting the spectrum from 'background' with the closest temperature.

    Parameters:
    - reaction_spectra: DataFrame with reaction data, indexed by datetime, with intensity columns for wavenumbers and a temperature column.
    - background_spectra: DataFrame with background spectra, indexed by datetime, with intensity columns and a temperature column.
    - temp_column: Name of the column indicating temperature in both DataFrames.
    - return_temp_column: If True, adds the temperature column to the corrected DataFrame.

    Returns:
    - corrected_reaction: DataFrame with background-corrected spectra.
    - temp_diffs: List of temperature differences between reaction and background spectra.
    """
    # Separate the intensity columns (wavenumbers) from exp_metadata by selecting only float columns
    reaction_intensities = reaction_spectra[[col for col in reaction_spectra.columns if isinstance(col, float)]]
    background_intensities = background_spectra[[col for col in reaction_spectra.columns if isinstance(col, float)]]

    # Create a new DataFrame for corrected spectra
    corrected_reaction = reaction_intensities.copy()

    temp_diffs = []  # Store the temperature differences for checking later

    # Iterate through each row in the reaction DataFrame (iterrow yields: index + row as a series)
    for idx, row in reaction_spectra.iterrows():
        current_temp = row[temp_column]

        # Find the closest temperature in the background DataFrame
        background_spectra = background_spectra.copy()  # Ensure a proper copy to avoid SettingWithCopyWarning
        background_spectra['temp_diff'] = np.abs(background_spectra[temp_column] - current_temp)
        closest_row = background_spectra.loc[background_spectra['temp_diff'].idxmin()]
        
        # Subtract the spectrum from the reaction spectrum
        corrected_reaction.loc[idx] = reaction_intensities.loc[idx] - background_intensities.loc[closest_row.name]

        # Add the temp diff column to a list to check if the temperatures are correct
        temp_diffs.append(closest_row['temp_diff'])

    if return_temp_column == True:
        # Add the temperature column to the corrected DataFrame
        corrected_reaction[temp_column] = reaction_spectra[temp_column]


    return corrected_reaction, temp_diffs

def linear_baseline_correction(spectra: pd.DataFrame, type: Literal['v1', 'v2'] = 'v1', area: tuple= (2600, 2400), temp_column: str = None):
    """
    Perform linear baseline correction on the spectra.

    Parameters:
    - spectra: DataFrame with the spectra to correct. Index should be filenames, columns are wavenumbers.
    - type: Type of baseline correction ('v1' or 'v2'). Default is 'v1'.
    - area: Range of wavenumbers for baseline correction. Default is (2600, 2400).
    - temp_column: Name of temperature column, if present.

    Returns:
    - corrected_spectra: DataFrame with the corrected spectra.
    """

    if temp_column is not None:
        spectra_noT = spectra.copy().drop(columns=[temp_column])  # Drop the temperature column if it exists
    else:
        spectra_noT = spectra.copy()

    spectra_noT.columns = spectra_noT.columns.astype(float)

    if type == 'v1':

        # BASELINE V1: select only the area between 2600 and 2400
        
        linear_baseline_v1 = spectra_noT.loc[:, area[0]:area[1]].mean(axis=1)

        spectra_v1_corrected = spectra_noT.copy()

        # subtract the baseline from each of the rows of the spectra
        for col in spectra_noT.columns:
            spectra_v1_corrected[col] = spectra_noT[col] - linear_baseline_v1


        # add the last column (the number of the spectrum) to the corrected spectra
        spectra_v1_corrected = spectra_v1_corrected.copy()

        if temp_column is not None:
            spectra_v1_corrected[temp_column] = spectra[temp_column]
        
        return spectra_v1_corrected

    elif type == 'v2':
        # Perform linear baseline correction using v2 method
        # BASELINE V2: use only a part of the spectra for a linear baseline between 1800 and 2500
        selected_area = spectra_noT.loc[:, area[0]:area[1]]

        linear_baseline_v2 = []
        for n in range(len(selected_area)):
            baseline = np.linspace(selected_area.iloc[n, 0], selected_area.iloc[n, -1], len(selected_area.columns))
            linear_baseline_v2.append(baseline)

        # do this for all rows
        for n in range(len(selected_area)):
            selected_area.iloc[n,:] = selected_area.copy().iloc[n,:] - linear_baseline_v2[n]

        spectra_v2_corrected = selected_area.copy()

        if temp_column is not None:
            spectra_v2_corrected[temp_column] = spectra[temp_column]

        return spectra_v2_corrected
    

def split_lightoff_lightout(data, temperatures = None):
    '''
    Split the spectra into lightoff and lightout datasets.

    Parameters:
    - data: DataFrame with the spectra to split (and temperature column if present).
    - temperatures: Optional list/Series of temperatures to split alongside spectra.

    Returns:
    - If temperatures is None: (lightoff_df, lightout_df)
    - If temperatures is provided: (lightoff_df, lightoff_temps, lightout_df, lightout_temps)
    '''

    DF_lightoff = data.iloc[:len(data)//2,:]
    DF_lightout = data.iloc[len(data)//2:,:]

    # Get the lightoff and lightout temperatures
    if temperatures is not None:
        lightoff_temperatures = temperatures[:len(temperatures)//2]
        lightout_temperatures = temperatures[len(temperatures)//2:]

        if lightoff_temperatures.iloc[-1] == lightout_temperatures.iloc[0]:
            print('Lightoff and lightout temperatures match')
        else:
            print('Lightoff and lightout temperatures DO NOT match, take care using this data')
    else:
        if DF_lightoff['T'].iloc[-1] == DF_lightout['T'].iloc[0]:
            print('Lightoff and lightout temperatures match')
        else:
            print('Lightoff and lightout temperatures DO NOT match, take care using this data')

    if temperatures is None:
        return DF_lightoff, DF_lightout
    else:
        return DF_lightoff, lightoff_temperatures, DF_lightout, lightout_temperatures

def import_gc_data_and_merge(gc_folder: str = 'GC/', DF_logfile = None):
    '''
    Import GC data from the specified path and merge with logfile.

    Parameters:
    - gc_folder: Path to the folder containing the GC data. Default is 'GC/'.
    - DF_logfile: DataFrame with the logfile data. If None, uses the global logfile variable.

    Returns:
    - gc_data: DataFrame with the GC data, indexed by datetime (optionally merged with logfile).
    '''
    try:
        gc_data_path = gc_folder + os.listdir(gc_folder)[0]
    except Exception as e:
        raise ValueError(f'Error finding GC data folder: {e}')
    time_difference = - pd.Timedelta(minutes = 5, seconds = 55) # GC pc is 5:55 minutes ahead of IR pc
    date_of_measurement = pd.to_datetime(str(DF_logfile['DateTime'].iloc[0])).strftime('%Y-%m-%d')

    # Load the GC data

    gc_data = pd.read_excel(gc_data_path, sheet_name='Sheet1', skiprows=3)
    gc_data.columns = [col.split('_')[0] for col in gc_data.columns]

    # put a time column
    gc_data['Time_GC'] = gc_data.iloc[:,0]

    # # drop the original column
    gc_data.drop(columns = gc_data.columns[0], inplace = True)

    # # make the time a string
    gc_data['Time_GC'] = gc_data['Time_GC'].apply(lambda x: x.strftime('%H:%M:%S'))


    # Create a datetime column that adjusts for midnight rollover
    gc_data['DateTime'] = pd.to_datetime(date_of_measurement + ' ' + gc_data['Time_GC'], format='%Y-%m-%d %H:%M:%S') + time_difference

    # Adjust for midnight rollover
    for i in range(1, len(gc_data)):
        if gc_data['DateTime'].iloc[i] < gc_data['DateTime'].iloc[i - 1]:
            gc_data.loc[i:, 'DateTime'] += pd.Timedelta(days=1)

    # Set the DateTime column as the index
    gc_data.index = gc_data['DateTime']
    gc_data.drop(columns=['DateTime'], inplace=True)

    # drop everything after the last measurement
    
    if merged_data is None:
        raise ValueError('No merged data found. Please run the merge_spectra_logfile function first.')

    gc_data = gc_data[gc_data.index <= merged_data.index[-1]]
    if gc_data.empty:
        raise ValueError('Time of GC data is not during measurement. Adjust date_of_measurement or time_difference')

    # fill all n.a. values with 0
    gc_data.fillna(0, inplace = True)

    pd.set_option('future.no_silent_downcasting', True)  # Enable downcasting warnings
    gc_data.replace('n.a.', 0, inplace = True)

    # add temperature column
    #gc_data['T'] = merged_data['Oven temp']

    # merge with logfile
    if DF_logfile is not None:
        
        # Make sure the DateTime column in the logfile is a pandas DateTime format
        DF_logfile['DateTime'] = pd.to_datetime(DF_logfile['DateTime'])

        try:
            gc_data_merged = pd.merge_asof(gc_data.sort_index(), DF_logfile.sort_values('DateTime'), on='DateTime', direction='nearest')
            gc_data_merged.index = gc_data_merged['DateTime']

            print('Found logfile, merging with GC data.')
            return gc_data_merged
        except Exception as e:
            print(f'Error during merging: {e}. Returning only GC data.')
            return gc_data
    else:
        print('No logfile provided. Returning only GC data.')
        return gc_data

# Define normalize_on_peak function to normalize the spectra on the peak maximum


# Define the peak ranges for the different bands
CO2_peak_range = (2370, 2355)
CO_peak_range = (2168, 2160)
gem1_peak_range = (2017, 2000)
gem2_peak_range = (2090, 2080)
lin_CO_Rh0_peak_range = (2060, 2055)
lin_CO_RhOx_peak_range = (2120, 2115)
CO2ads_peak_range = (2255, 2250)

def normalize_on_peak(DF_baselinecorrected, peak_range = (2370, 2360), temp = False):
    '''
    Normalize the spectra on the peak maximum within a given range.

    Parameters:
    - DF_baselinecorrected: DataFrame with baseline-corrected spectra. Index: filenames, columns: wavenumbers (+ optional temperature column).
    - peak_range: Tuple with wavenumber range for normalization. Default is (2370, 2360).
    - temp: Boolean, whether a temperature column is present. Default is True.

    Returns:
    - DF_baselinecorrected_normalized: DataFrame with normalized spectra.
    '''
    if temp == True:
        DF_baselinecorrected_noT = DF_baselinecorrected.copy()
        DF_baselinecorrected_noT = DF_baselinecorrected.iloc[:,:-1]
        DF_baselinecorrected_noT.columns = DF_baselinecorrected_noT.columns.astype(float)

    else:
        DF_baselinecorrected_noT = DF_baselinecorrected.copy()
        DF_baselinecorrected_noT.columns = DF_baselinecorrected_noT.columns.astype(float)

    DF_baselinecorrected_normalized = DF_baselinecorrected_noT/DF_baselinecorrected_noT.loc[:, peak_range[0] : peak_range[1]].max().max()

    DF_baselinecorrected_normalized.columns = DF_baselinecorrected_normalized.columns.astype(float)

    if temp == True:
        DF_baselinecorrected_normalized['T'] = DF_baselinecorrected['T']

    
    return DF_baselinecorrected_normalized

# Define get_maximum_peak function to get the maximum peak in a given range

def get_maximum_peak(DF_reaction_backgroundcorrected, peak_range, temp = False):
    '''
    Get the maximum peak value in a given wavenumber range.

    Parameters:
    - DF_reaction_backgroundcorrected: DataFrame with background-corrected spectra. Index: filenames, columns: wavenumbers (+ optional temperature column).
    - peak_range: Tuple with wavenumber range for peak search.
    - temp: Boolean, whether a temperature column is present. Default is True.

    Returns:
    - peak_maximum: Series with the maximum peak in the given range (indexed by temperature if temp=True).
    '''
    if temp == True:
        DF_reaction_backgroundcorrected_noT = DF_reaction_backgroundcorrected.copy()
        DF_reaction_backgroundcorrected_noT = DF_reaction_backgroundcorrected.iloc[:,:-1]
        DF_reaction_backgroundcorrected_noT.columns = DF_reaction_backgroundcorrected_noT.columns.astype(float)

    else:
        DF_reaction_backgroundcorrected_noT = DF_reaction_backgroundcorrected.copy()
        DF_reaction_backgroundcorrected_noT.columns = DF_reaction_backgroundcorrected_noT.columns.astype(float)

    peak_maximum = DF_reaction_backgroundcorrected_noT.loc[:,peak_range[0]:peak_range[1]].T.max()

    if temp == True:
        peak_maximum.index = DF_reaction_backgroundcorrected['T']

    return peak_maximum

Pt_experiments = r'D:\OneDrive - Universiteit Utrecht\Uni\PhD\Students\Tessa Bonneur\Pt-DRIFTS experiments/'
Rh_experiments = r'D:\OneDrive - Universiteit Utrecht\Uni\PhD\Students\Jakob Güldenberg\Research-Data-Jan\DRIFTS-Experiments/'

def load_experiment_metadata(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
        
def get_previous_data(experiment, datatype: str, convert_to_float: bool = False):
    '''
    Retrieve previous IR data from a specified experiment folder.

    Parameters:
    - prev_exp: Name of the previous experiment (string).
    - experiment_folder: Path to the folder containing experiment data (string).

    Returns:
    - spectra: DataFrame with the spectra data. The index should be the filenames and the columns should be the wavenumbers.
    '''
    all_metadata = load_experiment_metadata(r"D:\OneDrive - Universiteit Utrecht\Uni\PhD\Data\DRIFTS\DRIFTS_experiments_metadata.yaml")
    exp_metadata = all_metadata[experiment]
    exp_folder = exp_metadata['root'].replace("\\", '/') + exp_metadata['folder_name']

    try:
        DF = pd.read_csv(exp_folder + '/' + datatype + '.csv', index_col=0)
        print('Found previous ' + datatype + ' data for experiment ' + experiment + '.')
        if convert_to_float == True:
            DF.columns = DF.columns.astype(float)  # Convert column names to float if needed
            print('Converted column names to float for: ' + str(experiment) + ' ' + str(datatype))
        else:
            pass
        return DF
    except:
        print('File not found. Make sure there is a file named ' + datatype + ' in the folder')

def get_previous_GCdata(prev_exp, experiment_folder):
    '''
    Retrieve previous GC data from a specified experiment folder.

    Parameters:
    - prev_exp: Name of the previous experiment (string).
    - experiment_folder: Path to the folder containing experiment data (string).

    Returns:
    - gc_data: DataFrame with the GC data. The index should be the filenames and the columns should be the wavenumbers.
    '''
    file_to_open = 'No GC data found'

    for folder in os.listdir(experiment_folder):
        try:
            if folder.split(' ')[1] == prev_exp:
                folder_to_open = folder + '/'
        except: pass

    for file in os.listdir(experiment_folder+folder_to_open):
        if file.endswith('merged GC data.csv'):
            file_to_open = file
        else:
            pass

    try:
        return pd.read_csv(experiment_folder+folder_to_open+file_to_open, index_col=0)
    except:
        print(file_to_open + '. Make sure there is a file named ...merged GC data.csv in the folder')


def peak_height_plot(dataset_lightoff, dataset_lightout, normalization_range, peak_maximum_range, lightoff_temp, lightout_temp, title = None, xlim = None, ylim = None):
    """
    Plot the peak height vs temperature for lightoff and lightout spectra.

    Parameters:
    - dataset_lightoff: DataFrame with lightoff spectra.
    - dataset_lightout: DataFrame with lightout spectra.
    - normalization_range: Wavenumber range for normalization.
    - peak_maximum_range: Wavenumber range for peak maximum.
    - lightoff_temp: Temperature(s) for lightoff.
    - lightout_temp: Temperature(s) for lightout.
    - title: Optional plot title.
    - xlim: Optional x-axis limits.
    - ylim: Optional y-axis limits.

    Returns:
    - None. Displays and saves a plot of peak height vs temperature.
    """

    # make a scatter plot of temperature 1 vs the maximum of the peaks in the fresh dataset
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    # gives vlines at lightoff temperatures
    ax.vlines(lightoff_temp, -10, 10, linestyle='--', color='Darkred')

    # give
    # s vlines at lightout temperatures
    ax.vlines(lightout_temp, -10, 10, linestyle='--', color='Darkblue')


    # Left plot
    dataset = dataset_lightoff
    normalization = normalization_range
    peak_maximum = peak_maximum_range
    axis = 0

    ax.plot(dataset['T'], get_maximum_peak(normalize_on_peak(dataset, normalization), peak_maximum) - get_maximum_peak(normalize_on_peak(dataset, normalization), peak_maximum).min(), label = 'Heating', color = 'Red')

    dataset = dataset_lightout
    ax.plot(dataset['T'], get_maximum_peak(normalize_on_peak(dataset, normalization), peak_maximum) - get_maximum_peak(normalize_on_peak(dataset, normalization), peak_maximum).min(), label = 'Cooling', color = 'Blue')

    ax.legend(fontsize = 15, loc = 'upper right')
    ax.tick_params(labelsize=16)

    # set the xlim to the range of the dataset
    if xlim:
        ax.set_xlim(xlim)
    else:
        pass
    # set the ylim to the range of the dataset
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0,1)

    ax.grid(False)
    ax.set_xlabel('Temperature (\u00B0C)', fontsize = 15, labelpad = 5)
    ax.set_ylabel('Rel. peak height (a.u.)', fontsize = 15, labelpad = 6)
    ax.label_outer()
    ax.set_xticks([100, 200, 300, 400])



    if title:
        fig.savefig(title+'.png', dpi=300, bbox_inches='tight')


def save_lightoff(GC_lightoff = None, GC_lightout = None, IR_lightoff_CO = None, IR_lightout_CO = None, IR_lightoff_CO2 = None, IR_lightout_CO2 = None):    # call this filename
    '''
    Save lightoff and lightout temperatures to a CSV file.

    Parameters:
    - GC_lightoff: Lightoff temperature from GC (optional).
    - GC_lightout: Lightout temperature from GC (optional).
    - IR_lightoff_CO: Lightoff temperature from IR CO (optional).
    - IR_lightout_CO: Lightout temperature from IR CO (optional).
    - IR_lightoff_CO2: Lightoff temperature from IR CO2 (optional).
    - IR_lightout_CO2: Lightout temperature from IR CO2 (optional).

    Returns:
    - lightoff_temperatures: DataFrame with saved lightoff temperatures.
    '''
    filename = os.getcwd().split('\\')[-1]
    filename

    # export all lightoff temperature to a csv file
    lightoff_temperatures = pd.DataFrame({'Lightoff IR (CO2)': [IR_lightoff_CO2], 'Lightoff GC': [GC_lightoff], 'Lightout IR (CO2)': [IR_lightout_CO2], 'Lightout GC': [GC_lightout], 'Lightoff IR (CO)': [IR_lightoff_CO], 'Lightout IR (CO)': [IR_lightout_CO]}, index=[filename])
    lightoff_temperatures.to_csv(r'D:\OneDrive - Universiteit Utrecht\Uni\PhD\Data\Lightoff temperatures.csv')

    pd.read_csv(r'D:\OneDrive - Universiteit Utrecht\Uni\PhD\Data\Lightoff temperatures.csv', index_col=0)
    return lightoff_temperatures


def append_lightoff(experiment_name = None, GC_lightoff = None, GC_lightout = None, IR_lightoff_CO = None, IR_lightout_CO = None, IR_lightoff_CO2 = None, IR_lightout_CO2 = None):    # call this filename
    '''
    Append or update lightoff and lightout temperatures in the CSV file.

    Parameters:
    - filename: Name of the experiment/file.
    - GC_lightoff: Lightoff temperature from GC (optional).
    - GC_lightout: Lightout temperature from GC (optional).
    - IR_lightoff_CO: Lightoff temperature from IR CO (optional).
    - IR_lightout_CO: Lightout temperature from IR CO (optional).
    - IR_lightoff_CO2: Lightoff temperature from IR CO2 (optional).
    - IR_lightout_CO2: Lightout temperature from IR CO2 (optional).

    Returns:
    - lightoff_temperatures_new: DataFrame with updated lightoff temperatures.
    '''

    # read the file
    lightoff_temperatures = pd.read_csv(r'D:\OneDrive - Universiteit Utrecht\Uni\PhD\Data\Lightoff temperatures.csv', index_col=0)

    if IR_lightoff_CO == None and GC_lightoff == None and IR_lightout_CO == None and GC_lightout == None and IR_lightoff_CO2 == None and IR_lightout_CO2 == None:
        return lightoff_temperatures
    elif experiment_name is None:
        return lightoff_temperatures
    else:
        # check if the row name already exists
        if experiment_name in lightoff_temperatures.index:
            print('Entry already exists, overwriting...')
            # remove the row with the same name
            lightoff_temperatures = lightoff_temperatures.drop(experiment_name)
            print('Entry overwritten:', experiment_name)
        else:
            print('Entry does not exist, adding new row:', experiment_name)
        
        # append the new lightoff temperatures to the file
        lightoff_temperatures_new = pd.concat([lightoff_temperatures, pd.DataFrame({'Lightoff IR (CO2)': [IR_lightoff_CO2], 'Lightout IR (CO2)': [IR_lightout_CO2], 'Lightoff GC': [GC_lightoff], 'Lightout GC': [GC_lightout], 'Lightoff IR (CO)': [IR_lightoff_CO], 'Lightout IR (CO)': [IR_lightout_CO]}, index=[experiment_name])])
        lightoff_temperatures_new.sort_index(inplace=True)
        
        # sort the columns by name
        lightoff_temperatures_new = lightoff_temperatures_new.reindex(sorted(lightoff_temperatures_new.columns), axis=1)

        # export the file
        lightoff_temperatures_new.to_csv(r'D:\OneDrive - Universiteit Utrecht\Uni\PhD\Data\Lightoff temperatures.csv')

    return lightoff_temperatures_new


def plot_spectra_with_labels(spectra_to_plot: pd.DataFrame, labels_list: list, skip: int, title: str, colormap_string: Literal['combined', 'lightoff', 'lightout'], folder: str, xlim = None, ylim = None, figsize = (18,10), **kwargs):
    """
    Plot spectra with temperature labels and save as PNG.

    Parameters:
    - spectra_to_plot: DataFrame with spectra to plot.
    - labels_list: List of labels. Can be temperatures or other identifiers.
    - skip: Number of spectra to skip between each plot.
    - title: Plot title.
    - colormap_string: Colormap type ('combined', 'lightoff', 'lightout').
    - folder: Folder to save the plot.
    - xlim: Optional x-axis limits.
    - ylim: Optional y-axis limits.
    - figsize: Figure size tuple.
    - kwargs: Additional arguments for plotting.

    Returns:
    - None. Displays and saves the plot.
    """
    # Making the colormap
    if colormap_string == 'combined':
        colormap1 = plt.cm.Reds(np.linspace(0.3, 1, len(spectra_to_plot)//2 + 1))
        colormap2 = plt.cm.Blues(np.linspace(1, 0.3, len(spectra_to_plot)//2))
        colormap = np.vstack((colormap1, colormap2))

        if colormap.shape[0] != len(spectra_to_plot):
            print('Colormap shape does not match the number of spectra. Adjusting colormap.')
            colormap1 = plt.cm.Reds(np.linspace(0.3, 1, len(spectra_to_plot)//2 + 1))
            colormap2 = plt.cm.Blues(np.linspace(1, 0.3, len(spectra_to_plot)//2))
            colormap = np.vstack((colormap1, colormap2))

    elif colormap_string == 'lightoff':
        colormap = plt.cm.coolwarm(np.linspace(0, 1, len(spectra_to_plot)))

    elif colormap_string == 'lightout':
        colormap = plt.cm.coolwarm(np.linspace(1, 0, len(spectra_to_plot)))
    else:
        print('Colormap not recognized. Using default colormap.')
        colormap = plt.cm.coolwarm(np.linspace(0, 1, len(spectra_to_plot)))

    # Plotting the spectra
    fig, ax = plt.subplots(figsize = figsize)
    
    if labels_list is None:
        plot_labels = range(len(spectra_to_plot.index))
        legend_title = '#'
    else:
        plot_labels = labels_list
        legend_title = 'T (\u00B0C)'

    for n in np.arange(0, len(spectra_to_plot), skip):
        # Avoid deprecated Series.__getitem__ by using .iloc[n] for positional access
        ax.plot(spectra_to_plot.columns, spectra_to_plot.iloc[n], label=plot_labels[n], c=colormap[n], **kwargs)

    ax.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax.set_ylabel('Intensity')
    ax.set_title(title)
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    legend_labels = len(labels)

    if legend_labels <= 20:
        ax.legend(title = legend_title, loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize = 12)
    else:
        print('Too many labels ({} > 20), skipping over legend'.format(legend_labels))
        # Show only every nth label so that total labels <= 20
        n_skip = max(1, legend_labels // 20)
        # Use list slicing instead of get_item
        ax.legend(handles[::n_skip], [str(labels[i]) for i in range(0, len(labels), n_skip)], title=legend_title, loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize=12)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # save the figure with the title as filename
    fig.savefig(folder + '/' + title+'.png', dpi=300, bbox_inches='tight')

    plt.show()


def gas_analysis_plot(merged_data, folder, gas_flow = CO_flow):
    '''
    Plot temperature and gas flow over time and save as PNG.

    Parameters:
    - merged_data: DataFrame with merged spectra and logfile data.
    - folder: Folder to save the plot.
    - gas_flow: Column name for gas flow to plot (default: CO_flow).

    Returns:
    - None. Displays and saves the plot.
    '''

    # Plot temperature and gas flow over time
    fig, ax1 = plt.subplots(figsize=(12, 6))

    title = 'Temperature and gas flow overview'

    color = 'tab:red'
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Temperature (\u00B0C)', color=color, labelpad = 10)
    ax1.plot(merged_data['Number'], merged_data[Oven_temp], color=color, label='Temperature')
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Gas flow (mL/min)', color=color, labelpad = 10)
    ax2.plot(merged_data['Number'], merged_data[gas_flow], color=color, label='Gas Flow')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(str(Oven_temp) + ' (left) and '+ str(gas_flow) + ' (right)')
    plt.grid(alpha = 0.3)
    plt.show()

    fig.savefig(folder + '/' + title+'.png', dpi=300, bbox_inches='tight')

def plot_COox_conversion(gc_data_reaction, reaction_start, reaction_end, folder):
    """
    Plot GC data for CO conversion and CO2 formation during the reaction.

    Parameters:
    - gc_data_reaction: DataFrame with GC data for the reaction.
    - reaction_start: Start time of the reaction (pd.Timestamp).
    - reaction_end: End time of the reaction (pd.Timestamp).
    - folder: Folder to save the plot.

    Returns:
    - None. Displays and saves the plot.
    """

    gc_data_reaction.loc[:, 'X(CO)'] = 1 - (gc_data_reaction['CO'] / gc_data_reaction['CO'].max())
    gc_data_reaction.loc[:, 'X(CO) (%)'] = 100 * gc_data_reaction['X(CO)']

    gc_data_reaction.loc[:, 'X(CO2)'] = (gc_data_reaction['CO2'] / gc_data_reaction['CO2'].max())
    gc_data_reaction.loc[:, 'X(CO2) (%)'] = 100 * gc_data_reaction['X(CO2)']

    fig, ax = plt.subplots(figsize=(10, 5))

    gc_data_reaction.plot(x='Time',y='X(CO) (%)', ax=ax, title='GC data for CO conversion during reaction')
    gc_data_reaction.plot(x='Time',y='X(CO2) (%)',ax=ax, title='GC data for CO2 formation during reaction')

    ax.set_title('CO conversion and CO2 formation')
    ax.set_xlabel('Time')
    ax.set_ylabel('Conversion or formation (%)')

    fig.savefig(folder + '/' + 'GC data for CO conversion and CO2 formation during reaction.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_lightoff(gc_lightoff, gc_lightout, folder, gas: str = 'CO'):
    '''
    Plot lightoff and lightout curves for CO or CO2 and save as PNG.

    Parameters:
    - gc_lightoff: DataFrame with lightoff GC data.
    - gc_lightout: DataFrame with lightout GC data.
    - folder: Folder to save the plot.
    - gas: String, 'CO' or 'CO2' (default: 'CO').

    Returns:
    - None. Displays and saves the plot.
    '''
    fig, ax = plt.subplots(figsize = (10, 4.5), constrained_layout=True)

    if gas == 'CO':
        title = 'Lightoff curves X(' + gas + ')'
    elif gas == 'CO2':
        title = 'Lightoff curves C(' + gas + ')'

    if gas == 'CO':
        ax.plot(gc_lightoff[Oven_temp], gc_lightoff['X(CO) (%)'], 'o--', markersize = 5, c= 'red', label ='Light-off') 
        ax.plot(gc_lightout[Oven_temp], gc_lightout['X(CO) (%)'], 'o--', markersize = 5, c= 'blue', label = 'Light-out')
    elif gas == 'CO2':
        ax.plot(gc_lightoff[Oven_temp], gc_lightoff['X(CO2) (%)'], 'o--', markersize = 5, c= 'red', label ='Light-off') 
        ax.plot(gc_lightout[Oven_temp], gc_lightout['X(CO2) (%)'], 'o--', markersize = 5, c= 'blue', label = 'Light-out')


    # add a horizontal line at 50% conversion
    ax.axhline(y=50, color='grey', linestyle='--', label = '50% conversion')

    #ax.set_xlim(45, 400)
    ax.set_xlabel('Temperature (\u00B0C)', labelpad = 10)
    if gas == 'CO':
        ax.set_ylabel('X('+ gas + ') (%)', labelpad = 10)
    elif gas == 'CO2':
        ax.set_ylabel('C('+ gas + ') (%)', labelpad = 10)
    ax.legend(loc = 'lower right')
    ax.set_title(title)

    fig.savefig(folder + '/' + title+'.png', dpi=300, bbox_inches='tight')

def get_lightoff_lightout_temperatures(CO2_peakheights_lightoff, CO2_peakheights_lightout, CO_peakheights_lightoff, CO_peakheights_lightout):
    """
    Calculate lightoff and lightout temperatures for CO and CO2 peak heights.

    Parameters:
    - CO2_peakheights_lightoff: Series of CO2 peak heights during lightoff.
    - CO2_peakheights_lightout: Series of CO2 peak heights during lightout.
    - CO_peakheights_lightoff: Series of CO peak heights during lightoff.
    - CO_peakheights_lightout: Series of CO peak heights during lightout.

    Returns:
    - IR_lightoff_CO2: Averaged CO2 lightoff temperature.
    - IR_lightout_CO2: Averaged CO2 lightout temperature.
    - IR_lightoff_CO: Averaged CO lightoff temperature.
    - IR_lightout_CO: Averaged CO lightout temperature.
    """
    # CO2 lightoff
    lightoff_temperature_CO2_low = CO2_peakheights_lightoff[(CO2_peakheights_lightoff.values <= 0.5) & (CO2_peakheights_lightoff.index > 80) & (CO2_peakheights_lightoff.index < 350)].index[-1]
    lightoff_temperature_CO2_high = CO2_peakheights_lightoff[(CO2_peakheights_lightoff.values >= 0.5) & (CO2_peakheights_lightoff.index > 80) & (CO2_peakheights_lightoff.index < 350)].index[0]
    # CO2 lightout
    lightout_temperature_CO2_high = CO2_peakheights_lightout[(CO2_peakheights_lightout.values <= 0.5) & (CO2_peakheights_lightout.index > 80) & (CO2_peakheights_lightout.index < 350)].index[0]
    lightout_temperature_CO2_low = CO2_peakheights_lightout[(CO2_peakheights_lightout.values >= 0.5) & (CO2_peakheights_lightout.index > 80) & (CO2_peakheights_lightout.index < 350)].index[-1]
    # CO lightoff
    lightoff_temperature_CO_low = CO_peakheights_lightoff[(CO_peakheights_lightoff.values <= 0.5) & (CO_peakheights_lightoff.index > 80) & (CO_peakheights_lightoff.index < 350)].index[0]
    lightoff_temperature_CO_high = CO_peakheights_lightoff[(CO_peakheights_lightoff.values >= 0.5) & (CO_peakheights_lightoff.index > 80) & (CO_peakheights_lightoff.index < 350)].index[-1]
    # CO lightout
    lightout_temperature_CO_high = CO_peakheights_lightout[(CO_peakheights_lightout.values <= 0.5) & (CO_peakheights_lightout.index > 80) & (CO_peakheights_lightout.index < 350)].index[-1]
    lightout_temperature_CO_low = CO_peakheights_lightout[(CO_peakheights_lightout.values >= 0.5) & (CO_peakheights_lightout.index > 80) & (CO_peakheights_lightout.index < 350)].index[0]

    IR_lightoff_CO2 = round((lightoff_temperature_CO2_low + lightoff_temperature_CO2_high)/2)
    IR_lightout_CO2 = round((lightout_temperature_CO2_low + lightout_temperature_CO2_high)/2)
    IR_lightoff_CO = round((lightoff_temperature_CO_low + lightoff_temperature_CO_high)/2)
    IR_lightout_CO = round((lightout_temperature_CO_low + lightout_temperature_CO_high)/2)

    print(f'CO2 lightoff temperature is between {lightoff_temperature_CO2_low} and {lightoff_temperature_CO2_high} \u00B0C')
    print(f'Averaged, this is {round((lightoff_temperature_CO2_low + lightoff_temperature_CO2_high)/2)} \u00B0C')
    print(f'CO2 lightout temperature is between {lightout_temperature_CO2_low} and {lightout_temperature_CO2_high} \u00B0C')
    print(f'Averaged, this is {round((lightout_temperature_CO2_low + lightout_temperature_CO2_high)/2)} \u00B0C')
    print(f'CO lightoff temperature is between {lightoff_temperature_CO_low} and {lightoff_temperature_CO_high} \u00B0C')
    print(f'Averaged, this is {round((lightoff_temperature_CO_low + lightoff_temperature_CO_high)/2)} \u00B0C')
    print(f'CO lightout temperature is between {lightout_temperature_CO_low} and {lightout_temperature_CO_high} \u00B0C')
    print(f'Averaged, this is {round((lightout_temperature_CO_low + lightout_temperature_CO_high)/2)} \u00B0C\n')
    print(f'Average lightoff temperature is {round((lightoff_temperature_CO_low + lightoff_temperature_CO_high + lightoff_temperature_CO2_low + lightoff_temperature_CO2_high)/4)} \u00B0C')
    print(f'Average lightout temperature is {round((lightout_temperature_CO_low + lightout_temperature_CO_high + lightout_temperature_CO2_low + lightout_temperature_CO2_high)/4)} \u00B0C')

    return IR_lightoff_CO2, IR_lightout_CO2, IR_lightoff_CO, IR_lightout_CO