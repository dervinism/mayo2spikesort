import numpy as np
from typing import Dict, Union
from pprint import pprint

import spikeinterface as si
import spikeinterface.sorters as ss


def get_ks4_params(recording:si.BaseRecording, \
                   channel_distance:float=35.0, template_duration:float=0.002) \
    -> Dict[str, Union[str, int, float, np.inf, None, bool]]: # type: ignore
    '''
    params = get_ks4_params(recording:spikeinterface.core.BaseRecording, \
              channel_distance:float=35.0, template_duration:float=0.002

    Kilosort 4 paramaters are stored in this function. Use this function to get
    the base parameters and edit them within your script.

    Args:
        recorder (spikeinterface.core.NwbRecordingExtractor, required,
            positional): NWB recording exgtractor object.
        channel_distance (float, optional, keyword): float numeric scalar
            denoting minimal distance between recording channels on the probe
            (default=35.0).
        template_duration (float, optional, keyword): float numeric scalar
            denoting template duration in seconds (default=0.002).
    
    Returns:
        params (dict): a dictionary holding kilosort 4 parameters. Edit these
        values, if needed.
    '''

    sorter = 'kilosort4'
    params = dict()

    params['Th_learned'] = 8                      # Reduce this to increase the number of detected units (default=8)
    params['Th_single_ch'] = 6                    # Reduce this to increase the number of detected units (default=6)
    params['Th_universal'] =  9                   # Used in universal template calculations (default=9)
    params['acg_threshold'] = 0.2                 # Fraction of refractory period violations used to assign "good" units (default=0.2)
    params['artifact_threshold'] = np.inf         # Absolute values to zero out (default=np.inf)
    params['bad_channels'] = None                 # Can indicate bad channels based on the electrodes table of the NWB file (default=None)
    params['batch_size'] = 120000                 # Number of samples per batch of data (default=60000)
    params['binning_depth'] = 5                   # The size of the drift correction 2D histogram bin in microns (default=5)
    params['ccg_threshold'] = 0.25                # Fraction of refractory period violations that are allowed in the CCG compared to baseline; used to perform splits and merges (default=0.25)
    params['chunk_duration'] = '1s'
    params['clear_cache'] = False
    params['cluster_downsampling'] = 20           # Inverse fraction of nodes used as landmarks during clustering (default=20)
    params['delete_recording_dat'] = True
    params['dmin'] = channel_distance             # Vertical spacing between channels in microns (default=None). This is adjusted automatically by Kilosort but may have to be set manually (e.g., median distance) for irregualrly spaced contacts.
    params['dminx'] = channel_distance            # Horizontal spacing between channels in microns (default=32). This is adjusted automatically by Kilosort but may have to be set manually (e.g., median distance) for irregualrly spaced contacts.
    params['do_CAR'] = True                       # Whether to apply common average referencing (default=True)
    params['do_correction'] = True                # Presumably, whether to apply drift correction (default=True)    
    params['drift_smoothing'] = [0.5, 0.5, 0.5]   # Amount of gaussian smoothing to apply to the spatiotemporal drift estimation, for correlation, time (units of registration blocks), and y (units of batches) axes (default=[0.5, 0.5, 0.5])
    params['duplicate_spike_ms'] = 0.25           # Time in ms for which subsequent spikes from the same cluster are assumed to be artifacts (default=0.25). A value of 0 disables this step.
    params['fs'] = recording.sampling_frequency   # Sampling frequency of probe (default=30000)
    params['highpass_cutoff'] = 300               # Critical frequency (Hz) for highpass Butterworth filter applied to data (default=300).
    params['invert_sign'] = False
    params['keep_good_only'] = False
    params['max_channel_distance'] = channel_distance # Templates farther away than this from their nearest channel will not be used. Also limits distance between compared channels during clustering. This should be set based on the probe geometry (default=32).
    params['max_peels'] = 100                     # Number of iterations to do over each batch of data in the matching pursuit step. More iterations may detect more overlapping spikes (default=100).
    params['max_threads_per_worker'] = 1
    params['min_template_size'] = 10              # Standard deviation of the smallest, spatial envelope Gaussian used for universal templates (default=10). This is presumably in sample points.
    params['mp_context'] = None
    params['n_jobs'] = 1
    params['n_pcs'] = 6                           # Number of single-channel PCs to use for extracting spike features (only used if templates_from_data is True; default=6).
    params['n_templates'] = 6                     # Number of single-channel templates to use for the universal templates (only used if templates_from_data is True; default=6).
    params['nblocks'] = 0                         # Number of non-overlapping blocks for drift correction (default=1). For probes with fewer channels (around 64 or less) or with sparser spacing (around 50um or more between contacts), drift estimates are not likely to be accurate, so drift correction should be skipped by setting nblocks = 0.
    params['nearest_chans'] = min(10, recording.channel_ids.size) # Number of nearest channels to consider when finding local maxima during spike detection (default=10).
    params['nearest_templates'] = recording.channel_ids.size # Number of nearest spike template locations to consider when finding local maxima during spike detection (default=100).
    params['nskip'] = 25                          # Batch stride for computing whitening matrix (default=25).
    params['nt'] = round(recording.sampling_frequency*template_duration/2)*2 + 1 # The number of time samples used to represent spike waveforms (use an odd number; default=61).
    params['nt0min'] = None                       # Sample index for aligning waveforms, so that their minimum or maximum value happens here. Defaults to int(20 * settings['nt']/61).
    params['pool_engine'] = 'process'
    params['position_limit'] = 100                # Maximum distance (in microns) between channels that can be used to estimate spike positions in `postprocessing.compute_spike_positions` (default=100). This does not affect spike sorting, only how positions are estimated after sorting is complete.
    params['progress_bar'] = True
    params['save_extra_vars'] = False
    params['save_preprocessed_copy'] = False
    params['scale'] = None                        # Scaling factor to apply to data before all other operations. In most cases this should be left as None, but may be necessary for float32 data for example. If needed, `shift` and `scale` should be set such that data is roughly in the range -100 to +100.
    params['shift'] = None                        # Scalar shift to apply to data before all other operations. In most cases this should be left as None, but may be necessary for float32 data for example. If needed, `shift` and `scale` should be set such that data is roughly in the range -100 to +100.
    params['sig_interp'] = 20                     # Approximate spatial smoothness scale in units of microns (default=20).
    params['skip_kilosort_preprocessing'] = False
    params['template_sizes'] = 5                  # Number of sizes for universal spike templates (multiples of the min_template_size; default=5).
    params['templates_from_data'] = True          # Indicates whether spike shapes used in universal templates should be estimated from the data or loaded from the predefined templates (default=True).
    params['torch_device'] = 'cuda'
    params['use_binary_file'] = True
    params['whitening_range'] = recording.channel_ids.size # Number of nearby channels used to estimate the whitening matrix (default=32).
    params['x_centers'] = None                    # Number of x-positions to use when determining center points for template groupings. If None, this will be determined automatically by finding peaks in channel density (default).
    #params['data_dtype'] = 'int32'

    return params