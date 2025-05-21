import json
import numpy as np

from kilosort import run_kilosort
from pynwb import NWBHDF5IO, NWBFile


#nwb_file = r'C:\Users\m329786\Data\msel_02698_ICU03.nwb'
#binary_file = r'C:\Users\m329786\Data\spikeinterface_output\kilosort4\sorter_output_standalone\msel_02698_ICU03\msel_02698_ICU03_TimeSeries_32000_Hz.dat'
#robe_file = r'C:\Users\m329786\Data\spikeinterface_output\kilosort4\sorter_output_standalone\msel_02698_ICU03\probe_ks4.json'

binary_file = r'C:\Users\m329786\Data\spikeinterface_output\kilosort4\sorter_output_standalone\msel_02750_ICU02a_micro_data\msel_02750_ICU02a_micro_data.dat'
probe_file = r'C:\Users\m329786\Data\spikeinterface_output\kilosort4\sorter_output_standalone\msel_02750_ICU02a_micro_data\probe_ks4.json'

binary_format = 'int16'
get_true_sr = False

# Load settings from a json file
#with open("settings.json", "r") as file:
#    settings_file_data = json.load(file)

# Load parameters from the NWB file
if get_true_sr:
    with NWBHDF5IO(nwb_file, "r") as io: # type: ignore
        nwb_file_data = io.read()
        microelectrodeData = nwb_file_data.acquisition['TimeSeries_32000_Hz'] # type: ignore
        sampling_rate = microelectrodeData.rate
else:
    sampling_rate = 32000

settings = dict()
settings['filename'] = binary_file
settings['n_chan_bin'] = 16 
settings['Th_learned'] = 8                      # Reduce this to increase the number of detected units (default=8)
settings['Th_single_ch'] = 6                    # Reduce this to increase the number of detected units (default=6)
settings['Th_universal'] =  9                   # Used in universal template calculations (default=9)
settings['acg_threshold'] = 0.2                 # Fraction of refractory period violations used to assign "good" units (default=0.2)
settings['artifact_threshold'] = np.inf         # Absolute values to zero out (default=np.inf)
settings['batch_size'] = 120000                 # Number of samples per batch of data (default=60000)
settings['binning_depth'] = 5                   # The size of the drift correction 2D histogram bin in microns (default=5)
settings['ccg_threshold'] = 0.25                # Fraction of refractory period violations that are allowed in the CCG compared to baseline; used to perform splits and merges (default=0.25)
settings['clear_cache'] = False
settings['cluster_downsampling'] = 20           # Inverse fraction of nodes used as landmarks during clustering (default=20)
settings['dmin'] = 35.0                         # Vertical spacing between channels in microns (default=None). This is adjusted automatically by Kilosort but may have to be set manually (e.g., median distance) for irregualrly spaced contacts.
settings['dminx'] = 35.0                        # Horizontal spacing between channels in microns (default=32). This is adjusted automatically by Kilosort but may have to be set manually (e.g., median distance) for irregualrly spaced contacts.
settings['do_CAR'] = True                       # Whether to apply common average referencing (default=True)
settings['drift_smoothing'] = [0.5, 0.5, 0.5]   # Amount of gaussian smoothing to apply to the spatiotemporal drift estimation, for correlation, time (units of registration blocks), and y (units of batches) axes (default=[0.5, 0.5, 0.5])
settings['duplicate_spike_ms'] = 0.25           # Time in ms for which subsequent spikes from the same cluster are assumed to be artifacts (default=0.25). A value of 0 disables this step.
settings['fs'] = sampling_rate                  # Sampling frequency of probe (default=30000)
settings['highpass_cutoff'] = 300               # Critical frequency (Hz) for highpass Butterworth filter applied to data (default=300).
settings['invert_sign'] = False
settings['max_channel_distance'] = 35.0         # Templates farther away than this from their nearest channel will not be used. Also limits distance between compared channels during clustering. This should be set based on the probe geometry (default=32).
settings['max_peels'] = 100                     # Number of iterations to do over each batch of data in the matching pursuit step. More iterations may detect more overlapping spikes (default=100).
settings['min_template_size'] = 10              # Standard deviation of the smallest, spatial envelope Gaussian used for universal templates (default=10). This is presumably in sample points.
settings['n_pcs'] = 6                           # Number of single-channel PCs to use for extracting spike features (only used if templates_from_data is True; default=6).
settings['n_templates'] = 6                     # Number of single-channel templates to use for the universal templates (only used if templates_from_data is True; default=6).
settings['nblocks'] = 0                         # Number of non-overlapping blocks for drift correction (default=1). For probes with fewer channels (around 64 or less) or with sparser spacing (around 50um or more between contacts), drift estimates are not likely to be accurate, so drift correction should be skipped by setting nblocks = 0.
settings['nearest_chans'] = 4                   # Number of nearest channels to consider when finding local maxima during spike detection (default=10).
settings['nearest_templates'] = 16              # Number of nearest spike template locations to consider when finding local maxima during spike detection (default=100).
settings['nskip'] = 25                          # Batch stride for computing whitening matrix (default=25).
settings['nt'] = 65                             # The number of time samples used to represent spike waveforms (use an odd number; default=61).
settings['nt0min'] = None                       # Sample index for aligning waveforms, so that their minimum or maximum value happens here. Defaults to int(20 * settings['nt']/61).
settings['position_limit'] = 100                # Maximum distance (in microns) between channels that can be used to estimate spike positions in `postprocessing.compute_spike_positions` (default=100). This does not affect spike sorting, only how positions are estimated after sorting is complete.
settings['save_preprocessed_copy'] = False
settings['scale'] = None                        # Scaling factor to apply to data before all other operations. In most cases this should be left as None, but may be necessary for float32 data for example. If needed, `shift` and `scale` should be set such that data is roughly in the range -100 to +100.
settings['shift'] = None                        # Scalar shift to apply to data before all other operations. In most cases this should be left as None, but may be necessary for float32 data for example. If needed, `shift` and `scale` should be set such that data is roughly in the range -100 to +100.
settings['sig_interp'] = 20                     # Approximate spatial smoothness scale in units of microns (default=20).
settings['template_sizes'] = 5                  # Number of sizes for universal spike templates (multiples of the min_template_size; default=5).
settings['templates_from_data'] = True          # Indicates whether spike shapes used in universal templates should be estimated from the data or loaded from the predefined templates (default=True).
settings['whitening_range'] = 16                # Number of nearby channels used to estimate the whitening matrix (default=32).
settings['x_centers'] = None                    # Number of x-positions to use when determining center points for template groupings. If None, this will be determined automatically by finding peaks in channel density (default).
settings['data_dtype'] = binary_format

ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
    run_kilosort(settings=settings, probe_name=probe_file, \
                 save_preprocessed_copy=True, data_dtype=settings['data_dtype'])