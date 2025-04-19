import math


class Quantizers:
    def __init__(self, time_shift_bins=64, coord_bins=32, max_slider_repeats=4, spinner_duration_bins=16):
        self.time_shift_bins = time_shift_bins
        self.coord_bins = coord_bins
        self.max_slider_repeats = max_slider_repeats
        self.spinner_duration_bins = spinner_duration_bins

    def quantize_time_shift(self, time_shift):
        """Quantizes the time shift into bins using log scaling"""
        epsilon = 1
        log_val = math.log2(time_shift + epsilon)
        # Normalize the log value to the range of bins
        min_log_val = math.log2(epsilon)
        max_log_val = math.log2(3000 + epsilon)
        # Calculate the log-scaled value
        log_val = (log_val - min_log_val) / (max_log_val - min_log_val)
        # Scale linearly to the number of bins   
        bin_val = int(log_val * (self.time_shift_bins - 1))
        return min(max(bin_val, 0), self.time_shift_bins - 1)