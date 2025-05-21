import pandas as pd
import numpy as np
import math

from typing import Optional

from pynwb import NWBHDF5IO, NWBFile
from probeinterface import Probe, ProbeGroup, generate_tetrode


# Local functions
def load_nwb_data(data_path:str) -> pd.DataFrame:
    '''Local function to load an NWB file'''

    with NWBHDF5IO(data_path, "r") as io: # type: ignore
        nwb_file_data = io.read()
        electrode_table = nwb_file_data.electrodes.to_dataframe() # type: ignore
        micro_inds = electrode_table.isMicro.values.astype(int).astype(bool)
        electrode_table = electrode_table.iloc[micro_inds,:]
    return electrode_table


def correct_coordinates(electrode_table:pd.DataFrame) -> pd.DataFrame:
    '''Local function for correcting electrode channel coordinates'''

    # Correct channel coordinates (all channels should have unique coordinates)
    nChans = electrode_table.shape[0]
    for iChan in range(1, nChans-1):
        if electrode_table['x'].iloc[iChan] == electrode_table['x'].iloc[iChan-1] and \
            electrode_table['y'].iloc[iChan] == electrode_table['y'].iloc[iChan-1] and \
            electrode_table['z'].iloc[iChan] == electrode_table['z'].iloc[iChan-1]:  # Check if the coordinates are the same
            electrode_table['x'].iloc[iChan+2] = electrode_table['x'].iloc[iChan]
            electrode_table['x'].iloc[iChan] = electrode_table['x'].iloc[iChan+1]

    # Convert channel coordinates to microns
    electrode_table['x'] = electrode_table['x'] * 1e3
    electrode_table['y'] = electrode_table['y'] * 1e3
    electrode_table['z'] = electrode_table['z'] * 1e3

    return electrode_table


def create_tetrode_group(electrode_table:pd.DataFrame, \
                         tetrode_wire_radius:float=math.sqrt(35.0**2 + 35.0**2)/2) \
                        -> tuple[ProbeGroup, dict]:
    '''
    Local function for creating a tetrode probe group.
    Each tetrode group is based on each electrode group in the electrode table.
    The function returns a probegroup in the spikeInterface format and a probe
    definition in the Kilosort4 format.
    '''

    # Parameters
    nChans_per_tetrode = 4

    # Calculate probe centres 
    mean_loc = electrode_table.loc[:, ['x', 'y', 'z']].mean(axis=0).values
    tetrode_groups = electrode_table.group.unique()
    nTetrodes = tetrode_groups.size
    tetrode_centers = np.zeros((nTetrodes, 3))
    tetrode_distances = np.zeros((nTetrodes, 1))
    for iCenter in range(nTetrodes):
        tetrode_centers[iCenter, :] = electrode_table.loc[electrode_table.group == tetrode_groups[iCenter], ['x', 'y', 'z']].mean(axis=0).values
        tetrode_distances[iCenter] = math.dist(tetrode_centers[iCenter, :], mean_loc)
    tetrode_centers_0 = tetrode_centers - tetrode_centers.mean(axis=0)
    directions = np.sign(np.sign(tetrode_centers_0).sum(axis=1))

    # Set up the probe group
    #tetrode_distances = np.array([[300], [200], [100], [0]])
    nChans = electrode_table.shape[0]
    probegroup = ProbeGroup()
    for iTetrode in range(nTetrodes):
        tetrode = generate_tetrode(r=tetrode_wire_radius)
        tetrode.move([int((directions[iTetrode]*tetrode_distances[iTetrode])[0]), 0])
        probegroup.add_probe(tetrode)
    probegroup.set_global_device_channel_indices(np.arange(nChans))

    # Visualise the tetrode group
    #df = probegroup.to_dataframe()
    #print(df)
    #plot_probe_group(probegroup, with_contact_id=True, same_axes=True)

    # Set up the Kilosort 4 probe
    channel_ids = probegroup.get_global_device_channel_indices()
    channel_map = np.full((nChans,), np.nan)
    for iChan in range(nChans):
        channel_map[iChan] = channel_ids[iChan][1]
    xc = np.full((nChans,), np.nan)
    yc = np.full((nChans,), np.nan)
    kcoords = np.full((nChans,), np.nan)
    for iTetrode in range(nTetrodes):
        channel_indices = probegroup.probes[iTetrode].device_channel_indices
        contact_positions = probegroup.probes[iTetrode].contact_positions
        xc[channel_indices] = contact_positions[:,0]
        yc[channel_indices] = contact_positions[:,1]
        kcoords[channel_indices] = iTetrode
    
    ks4_probe = dict()
    ks4_probe['chanMap'] = channel_map
    ks4_probe['xc'] = xc
    ks4_probe['yc'] = yc
    ks4_probe['kcoords'] = kcoords
    ks4_probe['n_chan'] = nChans

    return probegroup, ks4_probe