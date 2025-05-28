import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import json
import h5py
import numpy as np
import pandas as pd
import math
import platform
import os
import pickle
from pynwb import NWBHDF5IO, NWBFile
from pprint import pprint
from tempfile import TemporaryDirectory

from local_functions import load_nwb_data, correct_coordinates, create_tetrode_group
from ks4_params import get_ks4_params
from sc2_params import get_sc2_params, get_sc2_params2
from ms5_params import get_ms5_params, get_ms5si_params
from klusta_params import get_klusta_params

import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.curation as scur
import spikeinterface.widgets as sw

from probeinterface.plotting import plot_probe, plot_probe_group

from kilosort.io import save_probe


# User input
sorter = 'kilosort4' # 'kilosort4', 'spykingcircus2', 'mountainsort5', or 'klusta'
save_binary = False
save_probe_file = False
visualise_data = False
attach_probe = False
locations_2D = True

# Common settings: I/O
data_path = '/mnt/c/Users/m329786/Data/msel_02750_ICU02a_micro_data.dat'
cache_path = '/mnt/c/Users/m329786/Data/spikeinterface_cache' # Extract microelectrode recording data and store it locally as a binary file to be used by spike sorters
docker_image = 'docker.io/library/ks4-with-spikeinterface'
si_output_folder = '/mnt/c/Users/m329786/Data/spikeinterface_output' # Output folder for spike sorting results. Should be on a local SSD.
probe_file = '/mnt/c/Users/m329786/Data/probe_ks4.json'

# Common settings: Parameters
template_duration = 0.002  # seconds
channel_distance = 35.0 # microns
typical_sr = 32000 # Hz
typical_nChans = 16
binary_format = 'int16'


# Check if the operating system is Linux (because Docker with CUDA is required)
if not platform.system() == 'Linux':
    raise Exception("The platform is not Linux. Please run this code on a Linux machine only.")

# Extract recording for spikesorting
cache_path = cache_path + '/' + os.path.basename(data_path)[0:-4]
if visualise_data:
    load_time_vector = True
else:
    load_time_vector = False
if data_path.endswith('.dat'):
    recording = si.read_binary(file_paths=data_path, sampling_frequency=typical_sr, \
                               num_channels=typical_nChans, dtype=binary_format)
    #recording = si.read_binary(file_paths=data_path, sampling_frequency=typical_sr, \
    #                           num_chan=typical_nChans, dtype=binary_format)
else:
    if sorter == 'klusta':
        recording = se.NwbRecordingExtractor(file_path=data_path, electrical_series_name='TimeSeries_32000_Hz', \
                                            load_time_vector=load_time_vector)
    else:
        recording = se.NwbRecordingExtractor(file_path=data_path, electrical_series_path='acquisition/TimeSeries_32000_Hz', \
                                            load_time_vector=load_time_vector, use_pynwb=True, \
                                            cache=True, stream_cache_path=cache_path)

# Visualise the data
if visualise_data:
    timestamps = recording.get_times()
    trace = recording.get_traces(segment_index=0)
    iChan = 0

    plt.ion()
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps[::10], trace[::10,iChan], label='Trace')
    plt.xlabel('Time (s)')
    plt.ylabel('V (uV)')
    plt.title('Trace plot')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Extract recording metadata
if not data_path.endswith('.dat'):
    electrode_table = load_nwb_data(data_path)
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#    print(electrode_table)

# Correct electrode channel coordinates, if non-unique
if not data_path.endswith('.dat'):
    electrode_table = correct_coordinates(electrode_table) # type: ignore

# Create a tetrode probe group and attach it to the recording
if not data_path.endswith('.dat'):
    tetrode_wire_radius = math.sqrt(channel_distance**2 + channel_distance**2)/2 # microns
    (probegroup, ks4_probe) = create_tetrode_group(electrode_table, tetrode_wire_radius) # type: ignore
    if attach_probe:
        recording = recording.set_probegroup(probegroup)
    else:
        if locations_2D:
            locations = np.column_stack((ks4_probe['xc'], \
                                        ks4_probe['yc']))
        else:
            locations = np.column_stack((electrode_table.x.values, \
                                         electrode_table.y.values, \
                                         electrode_table.z.values)) # type: ignore
        recording.set_property("location", locations)
        groups = ks4_probe['kcoords']
        recording.set_channel_groups(np.array(groups).astype(int))
else:
    with open(probe_file, "r") as file:
        ks4_probe = json.load(file)
        locations = np.column_stack((ks4_probe['xc'], ks4_probe['yc']))
        recording.set_property("location", locations)
        groups = ks4_probe['kcoords']
        recording.set_channel_groups(np.array(groups).astype(int))

# Split recordings based on probe
#recordings = recording.split_by(property="group")

# Set spikesorter parameters
if sorter == 'kilosort4':
    params = get_ks4_params(recording, channel_distance, template_duration) # type: ignore
    docker_image = True
elif sorter == 'spykingcircus2':
    #params = get_sc2_params(channel_distance, template_duration) # type: ignore
    params = get_sc2_params2(channel_distance, template_duration) # type: ignore
    docker_image = False
elif sorter == 'mountainsort5':
    params = get_ms5si_params(recording, channel_distance, template_duration) # type: ignore
    docker_image = False
elif sorter == 'klusta':
    params = get_klusta_params(recording, channel_distance, template_duration) # type: ignore
    docker_image = False
else:
    raise Exception("Unsupported sorter type.")

# Save a binary file
if save_binary:
    recording.save(format="binary", folder=cache_path, overwrite=True)

# Save the Kilosort 4 probe
if save_probe_file:
    if sorter == 'kilosort4':
        save_probe(ks4_probe, cache_path + '/probe_ks4.json')
    elif not data_path.endswith('.dat'):
        with open(cache_path + '/si_probegroup.pkl', 'wb') as f:
            pickle.dump(probegroup, f) # type: ignore

# Run the spikesorter
if not sorter == 'mountainsort5':
    recording = si.common_reference(recording)
si_output_folder_specific = si_output_folder + '/' + sorter + '/sorter_output/' + \
                            os.path.basename(data_path)[0:-4]
#sorting = ss.run_sorter(sorter_name=sorter, recording=recording, # type: ignore \
#                        folder=si_output_folder_specific, \
#                        docker_image=docker_image, verbose=True, \
#                        remove_existing_folder=True, **params) # type: ignore
sorting = ss.run_sorter_by_property(sorter_name=sorter, recording=recording, # type: ignore \
                                    working_folder=si_output_folder_specific, \
                                    docker_image=docker_image, verbose=True, \
                                    grouping_property='group', **params) # type: ignore
print(sorting)

# Export spikesorting results to Phy
if sorter == 'klusta':
    sorting.save(folder=si_output_folder_specific+'/sorting')
sorting = si.load(file_or_folder_or_dict=si_output_folder_specific+'/sorting')
    
si_analyzer_binary_specific = si_output_folder + '/' + sorter + '/analyzer_binary/' + \
                              os.path.basename(data_path)[0:-4]
analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording)
analyzer.compute([
    "random_spikes",
    "noise_levels",
    "templates",
    "template_similarity",
    "unit_locations",
    "spike_amplitudes",
    "principal_components",
    "correlograms",
    "waveforms"
    ]
)
analyzer.compute("quality_metrics", metric_names=["snr"])

si_phy_specific = si_output_folder + '/' + sorter + '/phy/' + os.path.basename(data_path)[0:-4]
sexp.export_to_phy(analyzer, si_phy_specific, verbose=True)