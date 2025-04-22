import math


class Quantizers:
    def __init__(self, vocab):
        self.vocab = vocab
        # Define ranges/parameters for quantization based on vocab or defaults
        self.time_shift_epsilon = 1.0 # Avoid log(0)
        # Estimate max log value based on a reasonable max time shift (10 seconds)
        self.max_log_time_shift = math.log2(10000 + self.time_shift_epsilon)
        self.min_log_time_shift = math.log2(self.time_shift_epsilon)

        self.max_spinner_duration_ms = 10000 # Max duration

        self.playfield_width = 512
        self.playfield_height = 384

    def quantize_time_shift(self, dt_ms):
        """Quantizes time difference using logarithmic scaling."""
        dt_ms = max(0, dt_ms) # Ensure non-negative
        log_val = math.log2(dt_ms + self.time_shift_epsilon)
        # Scale log value to fit number of bins
        scaled_log = (log_val - self.min_log_time_shift) / (self.max_log_time_shift - self.min_log_time_shift)
        bin_index = int(scaled_log * self.vocab.time_shift_bins)
        # Clamp to valid range [0, num_bins - 1]
        return max(0, min(bin_index, self.vocab.time_shift_bins - 1))
    
    def dequantize_time_shift(self, bin_val):
        """Dequantizes time difference from bin index."""
        # Scale back to log value
        scaled_log = (bin_val / self.vocab.time_shift_bins) * (self.max_log_time_shift - self.min_log_time_shift) + self.min_log_time_shift
        # Convert back to time difference
        return 2 ** scaled_log - self.time_shift_epsilon
    
    def quantize_coord(self, coord_norm, axis=None): # axis might be useful later
        """Quantizes normalized coordinate (0-1) using linear scaling."""
        if not (0.0 <= coord_norm <= 1.0):
            coord_norm = max(0.0, min(1.0, coord_norm)) # Clamp just in case
        if axis == "x":
            # Ensure index doesn't accidentally become num_bins due to coord_norm == 1.0
            bin_index = min(int(coord_norm * self.vocab.coord_x_bins), self.vocab.coord_x_bins - 1)
        elif axis == "y":
            bin_index = min(int(coord_norm * self.vocab.coord_y_bins), self.vocab.coord_y_bins - 1)
        else:
            raise ValueError("Invalid axis for quantization. Use 'x' or 'y'.")
        return bin_index
    
    def dequantize_coord(self, bin_val, axis=None):
        """Dequantizes coordinate from bin index."""
        if axis == "x":
            # Scale back to normalized value
            return bin_val / (self.vocab.coord_x_bins - 1)
        elif axis == "y":
            return bin_val / (self.vocab.coord_y_bins - 1)
        else:
            raise ValueError("Invalid axis for dequantization. Use 'x' or 'y'.")
    
    def quantize_slider_repeats(self, repeats):
        """Clamps slider repeats to the maximum defined bin."""
        return min(repeats, self.vocab.max_slider_repeats)

    def dequantize_slider_repeats(self, bin_val):
        """Dequantizes slider repeats from bin index."""
        return bin_val

    def quantize_spinner_duration(self, duration_ms):
        """Quantizes spinner duration using linear scaling."""
        duration_ms = max(0, duration_ms)
        scaled_duration = duration_ms / self.max_spinner_duration_ms
        bin_index = int(scaled_duration * self.vocab.spinner_duration_bins)
        # Clamp to valid range [0, num_bins - 1]
        return max(0, min(bin_index, self.vocab.spinner_duration_bins - 1))

    def dequantize_spinner_duration(self, bin_val):
        """Dequantizes spinner duration from bin index."""
        scaled_duration = (bin_val / self.vocab.spinner_duration_bins) * self.max_spinner_duration_ms
        return scaled_duration
    
    def quantize_relative_coord(self, d_coord, axis=None):
        """Quantizes relative coordinate change (dx/dy) by scaling and using absolute bins."""
        # Clamp delta to the defined range [-max_delta, +max_delta]
        clamped_delta = max(-self.vocab.max_relative_delta, min(d_coord, self.vocab.max_relative_delta))
        # Scale to [0, 1] range
        scaled_delta = (clamped_delta + self.vocab.max_relative_delta) / (2 * self.vocab.max_relative_delta)
        # Use the absolute coordinate quantizer
        return self.quantize_coord(scaled_delta, axis)
    
    def dequantize_relative_coord(self, bin_val, axis=None):
        """Dequantizes relative coordinate change from bin index."""
        # Scale back to the original range [-max_delta, +max_delta]
        scaled_delta = (bin_val / (self.vocab.coord_bins - 1)) * (2 * self.vocab.max_relative_delta) - self.vocab.max_relative_delta
        # Use the absolute coordinate dequantizer
        return self.dequantize_coord(scaled_delta, axis)