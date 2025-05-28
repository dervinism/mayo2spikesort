import numpy as np
#from typing import Dict, Union
from pprint import pprint

import spikeinterface as si
import spikeinterface.sorters as ss


def get_klusta_params(recording:si.BaseRecording, \
                      channel_distance:float=35.0, template_duration:float=0.002): \
    #-> Dict[str, Union[str, int, float, np.inf, None, bool, dict]]: # type: ignore
    '''
    params = get_klusta_params(recording:spikeinterface.core.BaseRecording, \
              channel_distance:float=35.0, template_duration:float=0.002

    klusta paramaters are stored in this function. Use this function
    to get the base parameters and edit them within your script. These
    parameters should be used within SpikeInterface.

    Args:
        recorder (spikeinterface.core.NwbRecordingExtractor, required,
            positional): NWB recording exgtractor object.
        channel_distance (float, optional, keyword): float numeric scalar
            denoting minimal distance between recording channels on the probe
            (default=35.0).
        template_duration (float, optional, keyword): float numeric scalar
            denoting template duration in seconds (default=0.002).
    
    Returns:
        params (dict): a dictionary holding klusta parameters. Edit
        these values, if needed.
    '''

    sorter = 'klusta'
    params = ss.get_default_sorter_params(sorter)
    print(sorter+"parameters:")
    pprint(params)

    params['chunk_duration'] = '1s'
    params['adjacency_radius'] = None
    params['detect_sign'] = 0 # Default=-1
    params['extract_s_after'] = int(round((template_duration*recording.sampling_frequency)*(2/3))) # Default=32
    params['extract_s_before'] = int(round((template_duration*recording.sampling_frequency)/3)) # Default=16
    #params['max_threads_per_worker'] = 1
    #params['mp_context'] = None
    params['n_features_per_channel'] = 3
    params['n_jobs'] = 1
    params['num_starting_clusters'] = 50
    params['pca_n_waveforms_max'] = 10000
    #params['pool_engine'] = 'process'
    params['progress_bar'] = True
    params['threshold_strong_std_factor'] = 5
    params['threshold_weak_std_factor'] = 2

    return params