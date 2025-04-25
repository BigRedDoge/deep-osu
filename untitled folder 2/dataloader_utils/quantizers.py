import math


class Quantizers:
    def __init__(self, vocab):
        self.vocab = vocab
        # Define ranges/parameters for quantization based on vocab or defaults
        self.time_shift_epsilon = 1.0 # Avoid log(0)
        # Estimate max log value based on a reasonable max time shift (20 seconds)
        self.max_log_time_shift = math.log2(10000 + self.time_shift_epsilon)
        self.min_log_time_shift = math.log2(self.time_shift_epsilon)

        self.max_spinner_duration_ms = 10000 # Max duration

        self.playfield_width = 512
        self.playfield_height = 384

    def quantize_time_shift(self, dt_ms):
        """Quantizes time difference using logarithmic scaling."""
        log_val = math.log2(dt_ms + self.time_shift_epsilon)
        scaled_log = (log_val - self.min_log_time_shift) / (self.max_log_time_shift - self.min_log_time_shift)
        bin_index = int(scaled_log * self.vocab.time_shift_bins)
        print(bin_index)
        bin_index = max(0, min(bin_index, self.vocab.time_shift_bins - 1))  # Clamp to valid range
        return bin_index
        """
        time_shifts = []
        while dt_ms > 0:
            print(dt_ms)
            # Quantize the current time shift
            log_val = math.log2(dt_ms + self.time_shift_epsilon)
            scaled_log = (log_val - self.min_log_time_shift) / (self.max_log_time_shift - self.min_log_time_shift)
            bin_index = int(scaled_log * self.vocab.time_shift_bins)
            print(bin_index)
            bin_index = max(0, min(bin_index, self.vocab.time_shift_bins - 1))  # Clamp to valid range
            time_shifts.append(bin_index)

            # Dequantize the bin to get the actual time shift value
            quantized_time = self.dequantize_time_shift(bin_index)
            # Subtract the quantized time from the remaining time
            dt_ms -= quantized_time
            if quantized_time == 0:
                break
        """
        return time_shifts
    
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
            norm_val = bin_val / (self.vocab.coord_x_bins - 1)
            # Convert to absolute coordinate
            return norm_val * self.playfield_width
        elif axis == "y":
            norm_val = bin_val / (self.vocab.coord_y_bins - 1)
            return norm_val * self.playfield_height
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
        clamped_delta = max(-self.vocab.slider_max_relative_delta, min(d_coord, self.vocab.slider_max_relative_delta))
        # Scale to [0, 1] range
        scaled_delta = (clamped_delta + self.vocab.slider_max_relative_delta) / (2 * self.vocab.slider_max_relative_delta)
        # Use the absolute coordinate quantizer
        return self.quantize_coord(scaled_delta, axis)
    
    def dequantize_relative_coord(self, bin_val, axis=None):
        """Dequantizes relative coordinate change from bin index."""
        # Scale back to the original range [-max_delta, +max_delta]
        scaled_delta = (bin_val / (self.vocab.coord_bins - 1)) * (2 * self.vocab.max_relative_delta) - self.vocab.max_relative_delta
        # Use the absolute coordinate dequantizer
        return self.dequantize_coord(scaled_delta, axis)
    
    def quantize_slider_duration(self, length):
        """Quantizes slider duration using linear scaling."""
        scaled_duration = length / self.playfield_width
        bin_index = int(scaled_duration * self.vocab.slider_duration_bins)
        return max(0, min(bin_index, self.vocab.slider_duration_bins - 1))
    
    def dequantize_slider_duration(self, bin_val):
        """Dequantizes slider duration from bin index."""
        scaled_duration = (bin_val / self.vocab.slider_duration_bins) * self.playfield_width
        return scaled_duration
    