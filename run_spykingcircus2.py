import json
import numpy as np
import os
from pprint import pprint

import spikeinterface.full as si
import spikeinterface.sorters as ss
import spikeinterface.exporters as sexp
from pynwb import NWBHDF5IO, NWBFile


binary_file = '/mnt/c/Users/m329786/Data/spikeinterface_output/spykingcircus2/sorter_output/msel_02750_ICU02a_micro_data/msel_02750_ICU02a_micro_data.dat'
probe_file = '/mnt/c/Users/m329786/Data/spikeinterface_output/spykingcircus2/sorter_output/msel_02750_ICU02a_micro_data/probe_ks4.json'
si_output_folder = '/mnt/c/Users/m329786/Data/spikeinterface_output' # Output folder for spike sorting results. Should be on a local SSD.

template_duration = 0.002  # seconds
channel_distance = 35.0 # microns
binary_format = 'int16'
get_true_sr = False
nChans = 16
docker_image = False

# Load parameters from the NWB file
if get_true_sr:
    with NWBHDF5IO(nwb_file, "r") as io: # type: ignore
        nwb_file_data = io.read()
        microelectrodeData = nwb_file_data.acquisition['TimeSeries_32000_Hz'] # type: ignore
        sampling_rate = microelectrodeData.rate
else:
    sampling_rate = 32000

# Load the binary for spikesorting
recording = si.read_binary(file_paths=binary_file, sampling_frequency=sampling_rate, \
                           num_channels=nChans, dtype=binary_format)

# Define recording channel locations
with open(probe_file, "r") as file:
    ks4_probe = json.load(file)
locations = np.column_stack((ks4_probe['xc'], ks4_probe['yc']))
recording.set_property("location", locations)

# Define sorter parameters
sorter = 'spykingcircus2'
params = ss.get_default_sorter_params(sorter)
print(sorter+"parameters:")
pprint(params)

params_general = dict() # A dictionary to describe how templates should be computed
params_general['ms_before'] = template_duration/2 # Duration of the first half of the template in ms (default=2)
params_general['ms_after'] = template_duration/2 # Duration of the second half of the template in ms (default=2)
params_general['radius_um'] = channel_distance*2 # The spatial width of the templates. By default, this value is read from the probe file.
params['general'] = params_general

params_sparsity = dict() # A dictionary to be passed to all the calls to sparsify the templates
params_sparsity['method'] = 'snr'
params_sparsity['amplitude_mode'] = 'peak_to_peak'
params_sparsity['threshold'] = 1
params['sparsity'] = params_sparsity

params_filtering = dict() # A dictionary for the high_pass filter used during preprocessing
params_filtering['freq_min'] = 150
params_filtering['freq_max'] = 7000
params_filtering['ftype'] = 'bessel'
params_filtering['filter_order'] = 2
params_filtering['margin_ms'] = 10
params['filtering'] = params_filtering

params_whitening = dict() # A dictionary for the whitening used during preprocessing
params_whitening['mode'] = 'local'
params_whitening['regularize'] = False
params['whitening'] = params_whitening

params_detection = dict() # A dictionary for the peak detection component
params_detection['peak_sign'] = 'neg'
params_detection['detect_threshold'] = 5
params['detection'] = params_detection

params_selection = dict() # A dictionary for the peak selection component
params_selection['method'] = 'uniform'
params_selection['n_peaks_per_channel'] = 5000
params_selection['min_n_peaks'] = 100000
params_selection['select_per_channel'] = False
params_selection['seed'] = 42
params['selection'] = params_selection

params['apply_motion_correction'] = False, # Boolean to specify whether circus 2 should apply motion correction to the recording or not
    
params['motion_correction']['preset'] = 'dredge_fast' # A dictionary to be provided if motion correction has to be performed (dense probe only)

params['merging']['max_distance_um'] = 50 # A dictionary to specify the final merging param to group cells after template matching (auto_merge_units)

params['clustering']['legacy'] = True # A dictionary for the clustering component. Default, graph_clustering is used

params['matching']['method'] = 'circus-omp-svd' # A dictionary for the matching component. Default circus-omp-svd. Use None to avoid matching

params['apply_preprocessing'] = True # Boolean to specify whether circus 2 should preprocess the recording or not. If yes, then high_pass filtering + common median reference + whitening"

#params['matched_filtering'] = True # Boolean to specify whether circus 2 should detect peaks via matched filtering (slightly slower)

params['cache_preprocessing']['mode'] = 'memory' # How to cache the preprocessed recording. Mode can be memory, file, zarr, with extra arguments. In case of memory (default), memory_limit will control how much RAM can be used. In case of folder or zarr, delete_cache controls if cache is cleaned after sorting
params['cache_preprocessing']['memory_limit'] = 0.5
params['cache_preprocessing']['delete_cache'] = True

params['multi_units_only'] = False # Boolean to get only multi units activity (i.e. one template per electrode)

params['job_kwargs']['n_jobs'] = 0.5 # A dictionary to specify how many jobs and which parameters they should used

params['seed'] = 42 # An int to control how chunks are shuffled while detecting peaks

params['debug'] = False # Boolean to specify if internal data structures made during the sorting should be kept for debugging

# Run the spikesorter
si_output_folder_specific = si_output_folder + '/' + sorter + '/sorter_output/' + \
                            os.path.basename(binary_file)[0:-4] + '/dump'
sorting = ss.run_sorter(sorter_name=sorter, recording=recording, # type: ignore \
                        folder=si_output_folder_specific, \
                        docker_image=docker_image, verbose=True, \
                        remove_existing_folder=False, **params) # type: ignore
print(sorting)

# Export spikesorting results to Phy
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

si_phy_specific = si_output_folder + '/' + sorter + '/phy/' + os.path.basename(binary_file)[0:-4]
sexp.export_to_phy(analyzer, si_phy_specific, verbose=True)