from beatmapparser.parser import BeatmapOsu
from beatmapparser.enums import HitObjectType
from beatmapparser.models import HitComboOsu, HitCircleOsu, SliderOsu, SpinnerOsu

from vocabulary import BeatmapVocabulary
from quantizers import Quantizers
"""
quantizers = Quantizers(time_shift_bins=64, coord_bins=32, max_slider_repeats=4, spinner_duration_bins=16)
vocab = BeatmapVocabulary(quantizers)
parser = BeatmapOsu("path/to/beatmap.osu")
"""

class OsuToToken:
    def __init__(self, osu_file_path, quantizers, vocab, max_seq_len=20480):
        self.parsed_osu = BeatmapOsu(osu_file_path)
        self.quantizers = quantizers
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        # Initialize the sequence with Start Of Sequence token
        self.token_id_sequence = [vocab.sos_id]
        self.tokenize_osu()

    def _append_token(self, token_str):
        """Helper function to append a token to the vocabulary."""
        if len(self.token_id_sequence) >= self.max_seq_len - 1: # Leave space for EOS
            print("Warning: Sequence length exceeded max_seq_len. Stopping tokenization.")
            return False
        
        token_id = self.vocab.get_id(token_str)
        if token_id is None:
            print(f"Token not found for: {token_str}")
        else:
            self.token_id_sequence.append(token_id)
        return True
    
    def tokenize_osu(self):
        # Keep track of the time of the last event to calculate shifts
        last_event_time_ms = 0 

        # Iterate through the hit objects in the parsed osu file
        for hit_object in self.parsed_osu.hit_objects:
            # Calculate the time shift
            current_event_time_ms = hit_object.start_time
            time_difference_ms = current_event_time_ms - last_event_time_ms
            time_difference_ms = max(0, time_difference_ms) # Ensure non-negative

            time_bin = self.quantizers.quantize_time_shift(time_difference_ms)
            time_token_str = f"TIME_SHIFT_{time_bin}"

            if not self._append_token(time_token_str): break

            # Handle different hit object types
            if hit_object.start_combo:
                if not self._append_token("NEW_COMBO"): break

            print(type(hit_object))
            if isinstance(hit_object, HitCircleOsu):
                if not self._append_token("PLACE_CIRCLE"): break
                norm_x = hit_object.position.x / 512.0
                norm_y = hit_object.position.y / 384.0
                x_bin = self.quantizers.quantize_coord(norm_x, 'x')
                y_bin = self.quantizers.quantize_coord(norm_y, 'y')
                if not self._append_token(f"COORD_X_{x_bin}"): break
                if not self._append_token(f"COORD_Y_{y_bin}"): break

            elif isinstance(hit_object, SliderOsu):
                slider_type = hit_object.curve_type
                if not self._append_token(f"START_SLIDER_{slider_type}"): break

                norm_x = hit_object.position.x / 512.0
                norm_y = hit_object.position.y / 384.0
                x_bin = self.quantizers.quantize_coord(norm_x, 'x')
                y_bin = self.quantizers.quantize_coord(norm_y, 'y')
                if not self._append_token(f"COORD_X_{x_bin}"): break
                if not self._append_token(f"COORD_X_{y_bin}"): break

                anchor_points = hit_object.curve_points
                for anchor in anchor_points:
                    self._append_token("ADD_SLIDER_ANCHOR")
                    dx_bin = self.quantizers.quantize_relative_coord(anchor.x, 'x')
                    dy_bin = self.quantizers.quantize_relative_coord(anchor.y, 'y')
                    if not self._append_token(f"COORD_X_{dx_bin}"): break
                    if not self._append_token(f"COORD_Y_{dy_bin}"): break

                repeats = hit_object.repeat_count
                repeat_bin = self.quantizers.quantize_slider_repeats(repeats)
                if not self._append_token(f"END_SLIDER_{repeat_bin}R"): break

            elif isinstance(hit_object, SpinnerOsu):
                if not self._append_token("PLACE_SPINNER"): break
                end_time = hit_object.end_time
                duration_ms = end_time - current_event_time_ms
                duration_bin = self.quantizers.quantize_spinner_duration(duration_ms)
                if not self._append_token(f"END_SPINNER_DUR{duration_bin}"): break
            else:
                print(f"Warning: Unknown object type encountered at time {current_event_time_ms}. Skipping.")
            
            last_event_time_ms = current_event_time_ms

        if len(self.token_id_sequence) < self.max_seq_len:
            eos_id = self.vocab.get_id(self.vocab.eos_token)
            if eos_id is not None:
                self.token_id_sequence.append(eos_id)
            else:
                print("Error: EOS token ID not found!")
                return None
        # If the loop broke because the sequence was full, replace the last token with EOS
        elif len(self.token_id_sequence) == self.max_seq_len:
            eos_id = self.vocab.get_id(self.vocab.eos_token)
            if eos_id is not None:
                self.token_id_sequence[-1] = eos_id
            else:
                print("Error: EOS token ID not found!")
                return None # Critical error
        
        # Pad the sequence to max_seq_len
        padding_needed = self.max_seq_len - len(self.token_id_sequence)
        if padding_needed > 0:
            self.token_id_sequence.extend([self.vocab.pad_id] * padding_needed)

        return self.token_id_sequence[:self.max_seq_len]
