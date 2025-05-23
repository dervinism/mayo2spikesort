import numpy as np
from typing import Dict, Union
from pprint import pprint

import spikeinterface as si
import spikeinterface.sorters as ss
import mountainsort5 as ms5


def get_ms5_params(recording:si.BaseRecording, \
                   channel_distance:float=35.0, template_duration:float=0.002) \
    -> ms5.schemes.Scheme2SortingParameters.Scheme2SortingParameters:
    '''
    params = get_ms5_params(recording:spikeinterface.core.BaseRecording, \
              channel_distance:float=35.0, template_duration:float=0.002

    mountainsort 5 paramaters are stored in this function. Use this function
    to get the base parameters and edit them within your script. These
    parameters should be used with standalone mountainsort 5.

    Args:
        recorder (spikeinterface.core.NwbRecordingExtractor, required,
            positional): NWB recording exgtractor object.
        channel_distance (float, optional, keyword): float numeric scalar
            denoting minimal distance between recording channels on the probe
            (default=35.0).
        template_duration (float, optional, keyword): float numeric scalar
            denoting template duration in seconds (default=0.002).
    
    Returns:
        params (dict): a dictionary holding mountainsort 5 parameters. Edit
        these values, if needed.
    '''

    sorter = 'mountainsort5'
    params = ss.get_default_sorter_params(sorter)
    print(sorter+"parameters:")
    pprint(params)

    params = ms5.Scheme2SortingParameters
    params.detect_sign = 0 # Default=-1
    params.detect_threshold = 5.5
    params.detect_time_radius_msec = 0.5
    params.freq_max = 6000
    params.freq_min = 300
    params.npca_per_channel = 3
    params.npca_per_subdivision = 10
    params.scheme1_detect_channel_radius = channel_distance*2
    params.scheme2_detect_channel_radius = channel_distance*2
    params.scheme2_max_num_snippets_per_training_batch = 200
    params.scheme2_phase1_detect_channel_radius = channel_distance*2
    params.scheme2_training_duration_sec = 300
    params.scheme2_training_recording_sampling_mode = 'uniform'
    params.scheme3_block_duration_sec = 1800
    params.snippet_T1 = round((template_duration*recording.sampling_frequency)/2)
    params.snippet_T2 = round((template_duration*recording.sampling_frequency)/2)
    params.snippet_mask_radius = channel_distance*2

    return params


def get_ms5si_params(recording:si.BaseRecording, \
                   channel_distance:float=35.0, template_duration:float=0.002) \
    -> Dict[str, Union[str, int, float, np.inf, None, bool, dict]]: # type: ignore
    '''
    params = get_ms5si_params(recording:spikeinterface.core.BaseRecording, \
              channel_distance:float=35.0, template_duration:float=0.002

    mountainsort 5 paramaters are stored in this function. Use this function
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
        params (dict): a dictionary holding mountainsort 5 parameters. Edit
        these values, if needed.
    '''

    sorter = 'mountainsort5'
    params = ss.get_default_sorter_params(sorter)
    print(sorter+"parameters:")
    pprint(params)

    params['chunk_duration'] = '1s'
    params['delete_temporary_recording'] = True
    params['detect_sign'] = 0 # Default=-1
    params['detect_threshold'] = 5.5
    params['detect_time_radius_msec'] = 0.5
    params['filter'] = True
    params['freq_max'] = 6000
    params['freq_min'] = 300
    params['max_threads_per_worker'] = 1
    params['mp_context'] = None
    params['n_jobs'] = 1
    params['npca_per_channel'] = 3
    params['npca_per_subdivision'] = 10
    params['pool_engine'] = 'process'
    params['progress_bar'] = True
    params['scheme'] = '2'
    params['scheme1_detect_channel_radius'] = channel_distance*2 # Default=150
    params['scheme2_detect_channel_radius'] = channel_distance*2 # Default=50
    params['scheme2_max_num_snippets_per_training_batch'] = 200
    params['scheme2_phase1_detect_channel_radius'] = channel_distance*2 # Default=200
    params['scheme2_training_duration_sec'] = 300
    params['scheme2_training_recording_sampling_mode'] = 'uniform'
    params['scheme3_block_duration_sec'] = 1800
    params['snippet_T1'] = round((template_duration*recording.sampling_frequency)/2) # Default=20
    params['snippet_T2'] = round((template_duration*recording.sampling_frequency)/2) # Default=20
    params['snippet_mask_radius'] = channel_distance*2 # Default=250
    params['whiten'] = True

    return params