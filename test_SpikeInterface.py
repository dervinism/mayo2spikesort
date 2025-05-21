import matplotlib.pyplot as plt
from pprint import pprint

import spikeinterface.full as si  # import core only
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.curation as scur
import spikeinterface.widgets as sw

from probeinterface.plotting import plot_probe



# Settings
# Enter here



# Tutorial here onwards
# Set global arguments for parallel processing
#global_job_kwargs = dict(n_jobs=4, chunk_duration="1s")
#si.set_global_job_kwargs(**global_job_kwargs)


# Download the MEArec dataset
# local_path = si.download_dataset(remote_path="mearec/mearec_test_10s.h5")
# Load data directly from https://gin.g-node.org/NeuralEnsemble/ephy_testing_data/src/master/mearec/mearec_test_10s.h5 and place it in /Users/dervinis.martynas/spikeinterface_datasets/ephy_testing_data/mearec
local_path = '/Users/dervinis.martynas/spikeinterface_datasets/ephy_testing_data/mearec/mearec_test_10s.h5'
recording, sorting_true = se.read_mearec(local_path)
print(recording)
print(sorting_true)


# Visualize the first 5 seconds of traces and raster plots
w_ts = sw.plot_traces(recording, time_range=(0, 5))
w_rs = sw.plot_rasters(sorting_true, time_range=(0, 5))


# Retrieve info from BaseRecording
channel_ids = recording.get_channel_ids()
fs = recording.get_sampling_frequency()
num_chan = recording.get_num_channels()
num_seg = recording.get_num_segments()

print("Channel ids:", channel_ids)
print("Sampling frequency:", fs)
print("Number of channels:", num_chan)
print("Number of segments:", num_seg)


# Retrieve info from BaseSorting
num_seg = recording.get_num_segments()
unit_ids = sorting_true.get_unit_ids()
spike_train = sorting_true.get_unit_spike_train(unit_id=unit_ids[0])

print("Number of segments:", num_seg)
print("Unit ids:", unit_ids)
print("Spike train of first unit:", spike_train)


# Visualise the ephys probe
probe = recording.get_probe()
print(probe)
_ = plot_probe(probe)


# Filter the recording and apply common median reference (CMR)
recording_cmr = recording
recording_f = si.bandpass_filter(recording, freq_min=300, freq_max=6000)
print(recording_f)
recording_cmr = si.common_reference(recording_f, reference="global", operator="median")
print(recording_cmr)

# this computes and saves the recording after applying the preprocessing chain
recording_preprocessed = recording_cmr.save(format="binary")
print(recording_preprocessed)


# Print available spikesorters and their parameters
print("Available sorters", ss.available_sorters())
print("Installed sorters", ss.installed_sorters())

print("Kilosort 4 parameters:")
pprint(ss.get_default_sorter_params("kilosort4"))
print("SpykingCircus 2 parameters:")
pprint(ss.get_default_sorter_params("spykingcircus2"))
#print("Mountainsort 5 parameters:") # Not installed. Try Docker image
#pprint(ss.get_default_sorter_params("mountainsort5"))
print("Tridesclous 2 parameters:")
pprint(ss.get_default_sorter_params("tridesclous2"))
print("Ironclust parameters:")
pprint(ss.get_default_sorter_params("ironclust"))


# Run one of the spikesorters: Kilosort 4
sorting_KS4 = ss.run_sorter(sorter_name="kilosort4", recording=recording_preprocessed, folder="spikesorting/kilosort4_output", verbose=True, remove_existing_folder=True)
print(sorting_KS4)

# Run one of the spikesorters: SpykingCircus 2
sorting_SC2 = ss.run_sorter(sorter_name="spykingcircus2", recording=recording_preprocessed, folder="spikesorting/spykingcircus2_output", verbose=True, remove_existing_folder=True)
print(sorting_SC2)

# Run one of the spikesorters: Tridesclous 2
other_params = ss.get_default_sorter_params("tridesclous2")
other_params["detect_threshold"] = 6
sorting_TDC2 = ss.run_sorter(sorter_name="tridesclous2", recording=recording_preprocessed, folder="spikesorting/tridesclous2_output", remove_existing_folder=True, **other_params)
print(sorting_TDC2)

# Run one of the spikesorters: Ironclust
sorting_IC = ss.run_sorter(sorter_name="ironclust", recording=recording_preprocessed, folder="spikesorting/ironclust_output", verbose=True, remove_existing_folder=True)
print(sorting_IC)



# Prevent closing of figures
plt.show(block=True)