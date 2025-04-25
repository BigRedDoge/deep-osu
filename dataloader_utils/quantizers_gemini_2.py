# dataloader_utils/quantizers.py
import math
import numpy as np # Using numpy for logspace/linspace

class QuantizationManager:
    def __init__(self, vocab):
        """
        Initializes the quantization manager.

        Args:
            vocab (BeatmapVocabulary): The vocabulary instance containing bin counts and ranges.
        """
        self.vocab = vocab

        # --- Define Ranges for Quantization ---
        # Time Shift (Example: Logarithmic, 1ms to 10 seconds)
        self.time_shift_min_ms = 1.0
        self.time_shift_max_ms = 10000.0
        # Precompute logspace edges for time shift for faster lookup
        # Add epsilon to max to ensure max value falls into last bin
        self.time_shift_log_edges = np.logspace(
            np.log10(self.time_shift_min_ms),
            np.log10(self.time_shift_max_ms + 1e-6), # Epsilon for inclusion
            num=self.vocab.time_shift_bins + 1 # Need N+1 edges for N bins
        )

        # Coordinates (Linear, 0.0 to 1.0 normalized)
        # Ranges are implicitly 0-1

        # Relative Coordinates (Linear, -max_delta to +max_delta pixels)
        # Range is defined by vocab.slider_max_relative_delta

        # Spinner Duration (Linear, e.g., 0ms to 10 seconds)
        self.spinner_min_ms = 0.0
        self.spinner_max_ms = 10000.0

        # --- NEW: Beat Length (Logarithmic, e.g., 30ms to 2000ms) ---
        # Corresponds roughly to 30 BPM to 2000 BPM
        self.beat_length_min_ms = 30.0
        self.beat_length_max_ms = 2000.0
        # Add epsilon for inclusion
        self.beat_length_log_edges = np.logspace(
            np.log10(self.beat_length_min_ms),
            np.log10(self.beat_length_max_ms + 1e-6),
            num=self.vocab.beat_length_bins + 1
        )

        # --- NEW: Slider Velocity (Linear, e.g., 0.1x to 4.0x) ---
        self.sv_min = 0.1
        self.sv_max = 4.0
        # Precompute linspace edges for SV
        self.sv_linear_edges = np.linspace(
            self.sv_min,
            self.sv_max + 1e-6, # Epsilon for inclusion
            num=self.vocab.slider_velocity_bins + 1
        )

    # --- Helper for finding bin index ---
    def _find_bin(self, value, edges):
        """Finds the bin index for a value given precomputed bin edges."""
        # np.searchsorted returns the index where value would be inserted to maintain order
        # 'right' side means values equal to an edge go into the bin to the right
        # Subtract 1 because searchsorted gives insertion point (0 to N), we want bin index (0 to N-1)
        bin_index = np.searchsorted(edges, value, side='right') - 1
        # Clamp index to valid range [0, num_bins - 1]
        num_bins = len(edges) - 1
        return max(0, min(bin_index, num_bins - 1))

    # --- Helper for getting bin midpoint ---
    def _get_bin_midpoint(self, bin_index, edges):
        """Gets the midpoint value of a given bin index."""
        if bin_index < 0 or bin_index >= len(edges) - 1:
            # Handle edge cases or invalid input if necessary
            return (edges[0] + edges[-1]) / 2 # Or raise error

        bin_start = edges[bin_index]
        bin_end = edges[bin_index + 1]
        # Use geometric mean for logarithmic scales, arithmetic for linear
        if np.all(edges > 0): # Check if likely log scale (all edges positive)
             # Avoid issues with log(0) if min_val is 0
             if bin_start <= 0: bin_start = 1e-9 # Small epsilon if start is 0 or less
             return math.sqrt(bin_start * bin_end) # Geometric mean
        else:
             return (bin_start + bin_end) / 2.0 # Arithmetic mean


    # --- Existing Quantization Methods (Updated to use helpers) ---

    def quantize_time_shift(self, dt_ms):
        """Quantizes time shift using precomputed logspace edges."""
        return self._find_bin(dt_ms, self.time_shift_log_edges)

    def quantize_coord(self, norm_val, axis):
        """Quantizes normalized coordinate (linear)."""
        bins = self.vocab.coord_x_bins if axis == 'x' else self.vocab.coord_y_bins
        # Linear quantization: scale 0-1 value to 0-bins range
        bin_index = int(norm_val * bins)
        # Clamp to handle norm_val == 1.0
        return max(0, min(bin_index, bins - 1))

    def quantize_relative_coord(self, delta, axis):
        """Quantizes relative coordinate delta (linear)."""
        bins = self.vocab.coord_x_bins if axis == 'x' else self.vocab.coord_y_bins
        max_delta = self.vocab.slider_max_relative_delta
        # Normalize delta from [-max_delta, +max_delta] to [0, 1]
        norm_01 = (delta + max_delta) / (2 * max_delta)
        # Clip normalized value to ensure it's within [0, 1]
        norm_01_clipped = max(0.0, min(norm_01, 1.0))
        bin_index = int(norm_01_clipped * bins)
        # Clamp to handle norm_01_clipped == 1.0
        return max(0, min(bin_index, bins - 1))

    def quantize_slider_repeats(self, repeats):
        """Quantizes slider repeats (usually just clamping)."""
        return max(0, min(repeats, self.vocab.max_slider_repeats))

    def quantize_spinner_duration(self, duration_ms):
        """Quantizes spinner duration (linear)."""
        norm_dur = (duration_ms - self.spinner_min_ms) / (self.spinner_max_ms - self.spinner_min_ms)
        norm_dur_clipped = max(0.0, min(norm_dur, 1.0))
        bin_index = int(norm_dur_clipped * self.vocab.spinner_duration_bins)
        return max(0, min(bin_index, self.vocab.spinner_duration_bins - 1))

    # --- NEW Quantization Methods ---

    def quantize_beat_length(self, beat_length_ms):
        """Quantizes beat length (ms) using precomputed logspace edges."""
        return self._find_bin(beat_length_ms, self.beat_length_log_edges)

    def quantize_slider_velocity(self, sv_multiplier):
        """Quantizes slider velocity multiplier using precomputed linear edges."""
        # Handle negative SV multipliers (inherited points) - map them to positive range
        # osu! encodes SV changes as negative values in timing points, e.g., -100 = 1.0x, -50 = 2.0x, -200 = 0.5x
        # The actual multiplier is 100 / (-value) if value < 0
        # We quantize the resulting positive multiplier (e.g., 0.1x to 4.0x)
        positive_multiplier = max(self.sv_min, min(sv_multiplier, self.sv_max)) # Clamp to defined range
        return self._find_bin(positive_multiplier, self.sv_linear_edges)

    # --- Existing Dequantization Methods (Updated to use helpers) ---

    def dequantize_time_shift(self, bin_index):
        """Dequantizes time shift bin index to approximate ms (geometric mean)."""
        return self._get_bin_midpoint(bin_index, self.time_shift_log_edges)

    def dequantize_coord(self, bin_index, axis):
        """Dequantizes coordinate bin index to normalized value (arithmetic mean)."""
        bins = self.vocab.coord_x_bins if axis == 'x' else self.vocab.coord_y_bins
        # Create temporary linear edges for dequantization
        linear_edges = np.linspace(0.0, 1.0, num=bins + 1)
        return self._get_bin_midpoint(bin_index, linear_edges)

    def dequantize_relative_coord(self, bin_index, axis):
        """Dequantizes relative coordinate bin index to pixel delta (arithmetic mean)."""
        bins = self.vocab.coord_x_bins if axis == 'x' else self.vocab.coord_y_bins
        max_delta = self.vocab.slider_max_relative_delta
        # Create temporary linear edges for the normalized [0, 1] range
        norm_edges = np.linspace(0.0, 1.0, num=bins + 1)
        # Get midpoint in normalized [0, 1] range
        norm_01 = self._get_bin_midpoint(bin_index, norm_edges)
        # Convert back to [-max_delta, +max_delta] range
        delta = (norm_01 * 2.0 - 1.0) * max_delta
        return delta

    def dequantize_slider_repeats(self, bin_index):
        """Dequantizes slider repeats bin index (identity)."""
        # Bin index directly corresponds to the number of repeats
        return max(0, min(bin_index, self.vocab.max_slider_repeats)) # Clamp just in case

    def dequantize_spinner_duration(self, bin_index):
        """Dequantizes spinner duration bin index to approximate ms (arithmetic mean)."""
        linear_edges = np.linspace(self.spinner_min_ms, self.spinner_max_ms, num=self.vocab.spinner_duration_bins + 1)
        return self._get_bin_midpoint(bin_index, linear_edges)

    # --- NEW Dequantization Methods ---

    def dequantize_beat_length(self, bin_index):
        """Dequantizes beat length bin index to approximate ms (geometric mean)."""
        return self._get_bin_midpoint(bin_index, self.beat_length_log_edges)

    def dequantize_slider_velocity(self, bin_index):
        """Dequantizes slider velocity bin index to multiplier (arithmetic mean)."""
        # Returns the positive multiplier (e.g., 1.5 for 1.5x)
        return self._get_bin_midpoint(bin_index, self.sv_linear_edges)

