import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


def nanvar(tensor, dim=None, keepdim=False):
    '''
    Compute the variance of a tensor, ignoring NaNs.
    '''
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor, dim=None, keepdim=False):
    '''
    Compute the standard deviation of a tensor, ignoring NaNs.
    '''
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


def interpolate_timeseries(
    timeseries: npt.NDArray, interp_length: int = 512
) -> npt.NDArray:
    '''
    Interpolate a timeseries to a new length.
    Works with both single channel [L] and multi-channel [C, L] inputs.
    '''
    # Handle both single and multi-channel cases
    if timeseries.ndim > 1:
        # Multi-channel case [C, L]
        num_channels, seq_length = original_shape
        x = np.linspace(0, 1, seq_length)
        x_new = np.linspace(0, 1, interp_length)
        
        # Reshape to handle all channels at once
        timeseries_reshaped = timeseries.reshape(num_channels, seq_length)
        result = np.zeros((num_channels, interp_length))
        
        # Create interpolation function for all channels at once
        f = interp1d(x, timeseries_reshaped, axis=1)
        result = f(x_new)
        
        return result
    else:
        original_shape = timeseries.shape
        # Single channel case [L]
        x = np.linspace(0, 1, original_shape[0])
        x_new = np.linspace(0, 1, interp_length)
        f = interp1d(x, timeseries)
        return f(x_new)


def upsample_timeseries(
    timeseries: npt.NDArray,  # [C, L]
    seq_len_channel: int,
    sampling_type: str = "pad",
    direction: str = "backward",
    **kwargs,
) -> npt.NDArray:
    '''
    Upsample a timeseries to a new length without using channel loops.
    '''
    num_channels, timeseries_len = timeseries.shape
    
    if sampling_type == "interpolate":
        # Use vectorized interpolation
        padded_timeseries = interpolate_timeseries(timeseries, seq_len_channel)
        # Create mask of ones with the same shape
        input_mask = np.ones((num_channels, seq_len_channel))
        
    elif sampling_type == "pad":
        if direction == "forward":
            # Create padding configuration for all channels at once
            pad_width = ((0, 0), (0, seq_len_channel - timeseries_len))
            padded_timeseries = np.pad(timeseries, pad_width, **kwargs)
            
            # Create mask - ones for original data, zeros for padding
            input_mask = np.ones((num_channels, seq_len_channel))
            input_mask[:, timeseries_len:] = 0
            
        elif direction == "backward":
            # Create padding configuration for all channels at once
            pad_width = ((0, 0), (seq_len_channel - timeseries_len, 0))
            padded_timeseries = np.pad(timeseries, pad_width, **kwargs)
            
            # Create mask - ones for original data, zeros for padding
            input_mask = np.ones((num_channels, seq_len_channel))
            input_mask[:, :seq_len_channel - timeseries_len] = 0
            
        else:
            error_msg = "Direction must be one of 'forward' or 'backward'"
            raise ValueError(error_msg)
    else:
        error_msg = "Sampling type must be one of 'interpolate' or 'pad'"
        raise ValueError(error_msg)
        
    assert padded_timeseries.shape[1] == seq_len_channel, "Padding failed"
    return padded_timeseries, input_mask


def downsample_timeseries(
    timeseries: npt.NDArray, seq_len: int, sampling_type: str = "interpolate"
): 
    '''
    TODO: still for a single channel now, need to extend to multiple channels
    Downsample a timeseries to a new length.
    '''
    input_mask = np.ones(seq_len)
    if sampling_type == "last":
        timeseries = timeseries[:seq_len]
    elif sampling_type == "first":
        timeseries = timeseries[seq_len:]
    elif sampling_type == "random":
        idx = np.random.randint(0, timeseries.shape[0] - seq_len)
        timeseries = timeseries[idx : idx + seq_len]
    elif sampling_type == "interpolate":
        timeseries = interpolate_timeseries(timeseries, seq_len)
    elif sampling_type == "subsample":
        factor = len(timeseries) // seq_len
        timeseries = timeseries[::factor]
        timeseries, input_mask = upsample_timeseries(
            timeseries, seq_len, sampling_type="pad", direction="forward"
        )
    else:
        error_msg = "Mode must be one of 'last', 'random',\
                'first', 'interpolate' or 'subsample'"
        raise ValueError(error_msg)
    return timeseries, input_mask