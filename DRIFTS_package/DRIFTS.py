import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

def quick_plot(spectra_to_plot: pd.DataFrame, skip: int=10, save = True):  # make a quick plot of the raw spectra and save it as a png file
    '''
    Make a quick plot of the raw spectra and save it as a png file.

    Parameters:
    - spectra_to_plot: DataFrame with the spectra to plot. The index should be the filenames and the columns should be the wavenumbers.
    - skip: Number of spectra to skip between each plot. Default is 10.
    - save: Boolean to save the plot as a png file. Default is True.

    Returns:
    - A quick plot of the raw spectra.
    '''


    colormap = plt.cm.coolwarm(np.linspace(1, 0, len(spectra_to_plot)))

    title = 'Raw spectra'


    fig, ax = plt.subplots(figsize=(18, 10))
    for n in np.arange(0,len(spectra_to_plot), skip):  # plotting from 1 onwards to skip the background spectrum
        ax.plot(spectra_to_plot.columns[:], spectra_to_plot.iloc[n,:], c=colormap[n])
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Intensity')
    ax.set_title(title)
    ax.set_xlim(spectra_to_plot.columns[0], spectra_to_plot.columns[-1])  # Invert x-axis

    if save == True:
        fig.savefig(title+'.png', dpi=300, bbox_inches='tight')

    plt.show()

def merge_spectra_logfile(spectra_path: str = 'data', logfile_path: str = 'data', setup: Literal['LP IR VMB', 'DeNOx'] = 'DeNOx', spectra_start_time: Literal['YYYY-MM-DD HH:MM:SS'] = None): # if spectra_start_time is None, it will use the first time in the logfile. 
    '''
    Merge spectra and logfile based on the start time of the logfile. If the spectra_start_time is not None, it will use that time as the start time for the spectra.

    Parameters:
    - logfile_path: Path to the folder containing the logfile OR DataFrame containing logfile data.
    - spectra_path: Path to the folder containing the spectra files OR DataFrame containing spectra data.
    - setup: Type of setup used for the experiment. Options are 'LP IR VMB' or 'DeNOx'.
    - spectra_start_time: Start time for the spectra. If None, it will use the first time in the logfile.

    Returns:
    - merged_data: DataFrame with the merged data, including the spectra and logfile information.
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
    - reaction: DataFrame with reaction data, indexed by datetime, with intensity columns for wavenumbers
      and a temperature column (e.g., 'Oven_temp').
    - background: DataFrame with spectra during inert rampdown, indexed by datetime, with intensity columns
      and a temperature column (e.g., 'Oven_temp').
    - temp_column: Name of the column indicating temperature in both DataFrames.

    Returns:
    - corrected_reaction: DataFrame with background-corrected spectra.
    - temp_diffs: List of temperature differences between reaction and background spectra.
    """
    # Separate the intensity columns (wavenumbers) from metadata by selecting only float columns
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
    Perform linear baseline correction on the specdtra.

    Parameters:
    - spectra: DataFrame with the specdtra to correct. The index should be the filenames and the columns should be the wavenumbers.
    - type: Type of baseline correction to perform. Options are 'v1' or 'v2'. Default is 'v1'.
    - area: Range of wavenumbers to use for the baseline correction. For v1, this is the part which is subtracted from everything, for v2 the specdtra are cut to that area. Default is (2600, 2400).

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
    Split the spectra into lightoff and lightout.

    Parameters:
    - spectra: DataFrame with the spectra to split and temperature column.

    Returns:
    - lightoff: DataFrame with the lightoff spectra.
    - lightout: DataFrame with the lightout spectra.
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
            raise ValueError('Lightoff and lightout temperatures DO NOT match, do not use this data')
    else:
        if DF_lightoff['T'].iloc[-1] == DF_lightout['T'].iloc[0]:
            print('Lightoff and lightout temperatures match')
        else:
            raise ValueError('Lightoff and lightout temperatures DO NOT match, do not use this data')

    if temperatures is None:
        return DF_lightoff, DF_lightout
    else:
        return DF_lightoff, DF_lightout, lightoff_temperatures, lightout_temperatures

def import_gc_data_and_merge(gc_folder: str = 'GC/', DF_logfile = None):
    '''
    Import the GC data from the specified path and merges with logfile. Very specific to the DeNOx setup and the way the data is saved.

    Parameters:
    - gc_folder: Path to the folder containing the GC data. Default is 'GC/'.
    - DF_logfile: DataFrame with the logfile data. Default is the global logfile variable.

    Returns:
        - gc_data: DataFrame with the GC data, indexed by datetime.
    '''
    try:
        gc_data_path = gc_folder + os.listdir(gc_folder)[0]
    except Exception as e:
        raise ValueError(f'Error finding GC data folder: {e}')
    time_difference = - pd.Timedelta(minutes = 5, seconds = 55) # GC pc is 5:55 minutes ahead of IR pc
    date_of_measurement = pd.to_datetime(DF_logfile.index[0]).strftime('%Y-%m-%d')

    # Load the GC data

    gc_data = pd.read_excel(gc_data_path, sheet_name='Sheet1', skiprows=3)
    gc_data.columns = [col.split('_')[0] for col in gc_data.columns]

    # put a time column
    gc_data['Time'] = gc_data.iloc[:,0]

    # # drop the original column
    gc_data.drop(columns = gc_data.columns[0], inplace = True)

    # # make the time a string
    gc_data['Time'] = gc_data['Time'].apply(lambda x: x.strftime('%H:%M:%S'))


    # Create a datetime column that adjusts for midnight rollover
    gc_data['DateTime'] = pd.to_datetime(date_of_measurement + ' ' + gc_data['Time'], format='%Y-%m-%d %H:%M:%S') + time_difference

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
        try:
            gc_data_merged = pd.merge_asof(gc_data.sort_values('DateTime'), DF_logfile.sort_values('DateTime'), on='DateTime', direction='nearest')
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

def normalize_on_peak(DF_baselinecorrected, peak_range = (2370, 2360), temp = True):
    '''
    Normalize the spectra on the peak maximum.

    Parameters:
    - DF_baselinecorrected: DataFrame with the baseline corrected spectra. The index should be the filenames and the columns should be the wavenumbers, and may include a temperature column at the end.
    - peak_range: Range of wavenumbers to use for the peak maximum. Default is (2370, 2360).

    Returns:
    - DF_baselinecorrected_normalized: DataFrame with the normalized spectra.
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

    DF_baselinecorrected_normalized['T'] = DF_baselinecorrected['T']

    
    return DF_baselinecorrected_normalized

# Define get_maximum_peak function to get the maximum peak in a given range

def get_maximum_peak(DF_reaction_backgroundcorrected, peak_range, temp = True):
    '''
    Get the maximum peak in a given range.

    Parameters:
    - DF_reaction_backgroundcorrected: DataFrame with the background corrected spectra. The index should be the filenames and the columns should be the wavenumbers, and may include a temperature column at the end.
    - peak_range: Range of wavenumbers to use for the peak maximum.

    Returns:
    - peak_maximum: Series with the maximum peak in the given range.
    '''
    if temp == True:
        DF_reaction_backgroundcorrected_noT = DF_reaction_backgroundcorrected.copy()
        DF_reaction_backgroundcorrected_noT = DF_reaction_backgroundcorrected.iloc[:,:-1]
        DF_reaction_backgroundcorrected_noT.columns = DF_reaction_backgroundcorrected_noT.columns.astype(float)

    else:
        DF_reaction_backgroundcorrected_noT = DF_reaction_backgroundcorrected.copy()
        DF_reaction_backgroundcorrected_noT.columns = DF_reaction_backgroundcorrected_noT.columns.astype(float)

    peak_maximum = DF_reaction_backgroundcorrected_noT.loc[:,peak_range[0]:peak_range[1]].T.max()
    peak_maximum.index = DF_reaction_backgroundcorrected['T']

    return peak_maximum

Pt_experiments = r'D:\OneDrive - Universiteit Utrecht\Uni\PhD\Students\Tessa Bonneur\Pt-DRIFTS experiments/'
Rh_experiments = r'D:\OneDrive - Universiteit Utrecht\Uni\PhD\Students\Jakob Güldenberg\Research-Data-Jan\DRIFTS-Experiments/'

def get_previous_IRdata(prev_exp, experiment_folder):
    '''
    Get the previous IR data from the specified folder. The folder should contain a subfolder with the name of the experiment.

    Parameters: 
    - prev_exp: Name of the previous experiment.
    - experiment_folder: Path to the folder containing the experiment data.

    Returns:
    - spectra: DataFrame with the spectra data. The index should be the filenames and the columns should be the wavenumbers.
    '''
    file_to_open = 'No data found for ' + prev_exp

    for folder in os.listdir(experiment_folder):
        try:
            if folder.split(' ')[1] == prev_exp:
                folder_to_open = folder + '/'
        except: pass

    for file in os.listdir(experiment_folder+folder_to_open):
        if file.endswith('background_corrected_spectra_T.csv'):
            file_to_open = file
        else:
            pass

    try:
        spectra = pd.read_csv(experiment_folder+folder_to_open+file_to_open, index_col=0)
#        spectra.columns = spectra.columns.astype(float)
        return spectra
    except:
        print(file_to_open + '. Make sure there is a file named ...background_corrected_spectra_T.csv in the folder')

def get_previous_GCdata(prev_exp, experiment_folder):
    '''
    Gets the previous GC data from the specified folder. The folder should contain a subfolder with the name of the experiment.

    Parameters:
    - prev_exp: Name of the previous experiment.
    - experiment_folder: Path to the folder containing the experiment data.

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


def peak_height_plot(dataset_lightoff, dataset_lightout, normalization_range, peak_maximum_range, lightoff_temp, lightout_temp, title = None):
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
    ax.set_ylim(0, 1)
    # ax.grid(False)
    ax.set_xlabel('Temperature (°C)', fontsize = 15, labelpad = 5)
    ax.set_ylabel('Rel. peak height (a.u.)', fontsize = 15, labelpad = 6)
    ax.label_outer()
    ax.set_xticks([100, 200, 300, 400])

    if title:
        fig.savefig(title+'.png', dpi=300, bbox_inches='tight')


def save_lightoff(lightoff_IR, lightoff_GC, lightout_IR, lightout_GC):    # call this filename
    filename = os.getcwd().split('\\')[-1]
    filename

    # export all lightoff temperature to a csv file
    lightoff_temperatures = pd.DataFrame({'Lightoff IR': [lightoff_IR], 'Lightoff GC': [lightoff_GC], 'Lightout IR': [lightout_IR], 'Lightout GC': [lightout_GC]}, index=[filename])
    lightoff_temperatures.to_csv(r'D:\OneDrive - Universiteit Utrecht\Uni\PhD\Data\Lightoff temperatures.csv')

    pd.read_csv(r'D:\OneDrive - Universiteit Utrecht\Uni\PhD\Data\Lightoff temperatures.csv', index_col=0)
    return lightoff_temperatures


def append_lightoff(lightoff_IR = None, lightoff_GC = None, lightout_IR = None, lightout_GC = None):    # call this filename
    

    # call this filename
    filename = os.getcwd().split('\\')[-1]

    # read the file
    lightoff_temperatures = pd.read_csv(r'D:\OneDrive - Universiteit Utrecht\Uni\PhD\Data\Lightoff temperatures.csv', index_col=0)

    if lightoff_IR == None and lightoff_GC == None and lightout_IR == None and lightout_GC == None:
        return lightoff_temperatures
    else:
        # check if the row name already exists
        if filename in lightoff_temperatures.index:
            print('Row already exists, overwriting')
            # remove the row with the same name
            lightoff_temperatures = lightoff_temperatures.drop(filename)

        else:
            print('Row does not exist, adding new row')
        
        # append the new lightoff temperatures to the file
        lightoff_temperatures_new = pd.concat([lightoff_temperatures, pd.DataFrame({'Lightoff IR': [lightoff_IR], 'Lightoff GC': [lightoff_GC], 'Lightout IR': [lightout_IR], 'Lightout GC': [lightout_GC]}, index=[filename])])
        lightoff_temperatures_new.sort_index(inplace=True)
        
        # export the file
        lightoff_temperatures_new.to_csv(r'D:\OneDrive - Universiteit Utrecht\Uni\PhD\Data\Lightoff temperatures.csv')

    return lightoff_temperatures_new


def plot_spectra_temperatures(spectra_to_plot, temperatures, skip, title, colormap_string, folder, xlim = None, ylim = None, figsize = (18,10), **kwargs):
    """
    Plot the spectra with the temperatures as labels.
    Parameters:
    - spectra_to_plot: DataFrame with the spectra to plot. The index should be the filenames and the columns should be the wavenumbers.
    - temperatures: List of temperatures to use as labels.
    - skip: Number of spectra to skip between each plot. Default is 10.
    - title: Title of the plot.
    - colormap_string: String to define the colormap. Options are 'coolwarm', 'combined', 'lightoff', 'lightout'.
    - folder: Folder to save the plot.
    - kwargs: Additional arguments to pass to the plot function.
    Returns:
    - A plot of the spectra with the temperatures as labels.
    """

    if colormap_string == 'combined':
        colormap1 = plt.cm.Reds(np.linspace(0.3, 1, len(spectra_to_plot)//2))
        colormap2 = plt.cm.Blues(np.linspace(1, 0.3, len(spectra_to_plot)//2))
        colormap = np.vstack((colormap1, colormap2))

    elif colormap_string == 'lightoff':
        colormap = plt.cm.coolwarm(np.linspace(0, 1, len(spectra_to_plot)))

    elif colormap_string == 'lightout':
        colormap = plt.cm.coolwarm(np.linspace(1, 0, len(spectra_to_plot)))
    else:
        print('Colormap not recognized. Using default colormap.')
        colormap = plt.cm.coolwarm(np.linspace(0, 1, len(spectra_to_plot)))

    fig, ax = plt.subplots(figsize = figsize)
    
    for n in np.arange(0, len(spectra_to_plot), skip):  # plotting until -1 to skip the T column
        ax.plot(spectra_to_plot.columns[:-1], spectra_to_plot.iloc[n,:-1], label=temperatures[n], c=colormap[n], **kwargs)

    ax.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax.set_ylabel('Intensity')
    ax.set_title(title)


# make a legend with a subset of the data

    if len(ax.get_legend_handles_labels()[1]) < 50:
        ax.legend(title = 'T (°C)', loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize = 12)

    else:
        print('Too many labels, legend disabled')
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

# save the figure with the title as filename
    fig.savefig(folder + '/' + title+'.png', dpi=300, bbox_inches='tight')

    plt.show();


def gas_analysis_plot(merged_data, gas_flow = CO_flow):

    # Plot temperature and gas flow over time
    fig, ax1 = plt.subplots(figsize=(12, 6))

    title = 'Temperature and gas flow overview'

    color = 'tab:red'
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Temperature (°C)', color=color, labelpad = 10)
    ax1.plot(merged_data['Number'], merged_data[Oven_temp], color=color, label='Temperature')
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Gas flow (mL/min)', color=color, labelpad = 10)
    ax2.plot(merged_data['Number'], merged_data[CO_flow], color=color, label='Gas Flow')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(str(Oven_temp) + ' (left) and '+ str(gas_flow) + ' (right)')
    plt.grid(alpha = 0.3)
    plt.show()

    fig.savefig(title + '.png', dpi=300, bbox_inches='tight')