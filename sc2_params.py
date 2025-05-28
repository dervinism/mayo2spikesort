import numpy as np
#from typing import Dict, Union
from pprint import pprint

import spikeinterface.sorters as ss


def get_sc2_params(channel_distance:float=35.0, template_duration:float=0.002): \
    #-> Dict[str, Union[str, int, float, np.inf, None, bool, dict]]: # type: ignore
    '''
    params = get_sc2_params(channel_distance:float=35.0, template_duration:float=0.002)

    spyKING CIRCUS 2 paramaters are stored in this function. Use this function
    to get the base parameters and edit them within your script.

    Args:
        channel_distance (float, optional, keyword): float numeric scalar
            denoting minimal distance between recording channels on the probe
            (default=35.0).
        template_duration (float, optional, keyword): float numeric scalar
            denoting template duration in seconds (default=0.002).
    
    Returns:
        params (dict): a dictionary holding spyking circus 2 parameters. Edit
        these values, if needed.
    '''

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
    
    return params


def get_sc2_params2(channel_distance:float=35.0, template_duration:float=0.002): \
    #-> Dict[str, Union[str, int, float, np.inf, None, bool, dict]]: # type: ignore
    '''
    params = get_sc2_params2(channel_distance:float=35.0, template_duration:float=0.002)

    spyKING CIRCUS 2 paramaters are stored in this function. Use this function
    to get the base parameters and edit them within your script.

    Args:
        channel_distance (float, optional, keyword): float numeric scalar
            denoting minimal distance between recording channels on the probe
            (default=35.0).
        template_duration (float, optional, keyword): float numeric scalar
            denoting template duration in seconds (default=0.002).
    
    Returns:
        params (dict): a dictionary holding spyking circus 2 parameters. Edit
        these values, if needed.
    '''

    sorter = 'spykingcircus2'
    params = ss.get_default_sorter_params(sorter)
    print(sorter+"parameters:")
    pprint(params)

    params['apply_motion_correction'] = False #True
    params['apply_preprocessing'] = True
    params['cache_preprocessing']['delete_cache'] = True
    params['cache_preprocessing']['memory_limit'] = 0.5
    params['cache_preprocessing']['mode'] = 'memory'
    params['clustering']['method'] = 'circus'
    params['clustering']['method_kwargs'] = {}
    params['debug'] = False
    params['detection']['method'] = 'matched_filtering'
    params['detection']['method_kwargs']['detect_threshold'] = 5
    params['detection']['method_kwargs']['peak_sign'] = 'neg' # Default='neg'
    params['filtering']['filter_order'] = 2
    params['filtering']['freq_max'] = 7000
    params['filtering']['freq_min'] = 150
    params['filtering']['ftype'] = 'bessel'
    params['filtering']['margin_ms'] = 10
    params['general']['ms_after'] = 1 # Default=2
    params['general']['ms_before'] = 1 # Default=2
    params['general']['radius_um'] = channel_distance*2 # Default=100
    params['job_kwargs']['n_jobs'] = 0.75
    params['matching']['method'] = 'circus-omp-svd'
    params['matching']['method_kwargs'] = {}
    params['merging']['max_distance_um'] = channel_distance*2 # Default=50
    params['motion_correction']['preset'] = 'dredge_fast'
    params['multi_units_only'] = False
    params['seed'] = 42
    params['selection']['method'] = 'uniform'
    params['selection']['method_kwargs']['min_n_peaks'] = 100000
    params['selection']['method_kwargs']['n_peaks_per_channel'] = 5000
    params['selection']['method_kwargs']['select_per_channel'] = False
    params['sparsity']['amplitude_mode'] = 'peak_to_peak'
    params['sparsity']['method'] = 'snr'
    params['sparsity']['threshold'] = 0.25
    params['templates_from_svd'] = True
    params['whitening']['mode'] = 'local'
    params['whitening']['regularize'] = False
    
    return params