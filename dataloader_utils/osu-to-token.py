from dataloader_utils.beatmapparser.parser import BeatmapOsu
from dataloader_utils.beatmapparser.enums import HitObjectType
from dataloader_utils.beatmapparser.models import HitComboOsu, HitCircleOsu, SliderOsu, SpinnerOsu

from dataloader_utils.vocabulary import BeatmapVocabulary
from dataloader_utils.quantizers import Quantizers

quantizers = Quantizers(time_shift_bins=64, coord_bins=32, max_slider_repeats=4, spinner_duration_bins=16)
vocab = BeatmapVocabulary(quantizers)
parser = BeatmapOsu("path/to/beatmap.osu")


class OsuToToken:
    def __init__(self, osu_file_path, quantizers, vocab):
        self.parsed_osu = BeatmapOsu(osu_file_path)
        self.quantizers = quantizers
        self.vocab = vocab

    def _append_token(self, token_str):
        """Helper function to append a token to the vocabulary."""
        token_id = self.vocab.get_id(token_str)
        if token_id is None:
            print(f"Token not found for: {token_str}")
        else:
            self.token_id_sequence.append(token_id)

    def tokenize_osu(self, parsed_osu, vocab, quantizers):
        # Initialize the sequence with Start Of Sequence token
        token_id_sequence = [vocab.sos_id]

        # Keep track of the time of the last event to calculate shifts
        last_event_time_ms = 0 

        # Iterate through the hit objects in the parsed osu file
        for hit_object in parsed_osu.hit_objects:
            # Calculate the time shift
            current_event_time_ms = hit_object.start_time
            time_difference_ms = current_event_time_ms - last_event_time_ms
            time_difference_ms = max(0, time_difference_ms) # Ensure non-negative

            time_bin = quantizers.quantize_time_shift(time_difference_ms)
            time_token_str = f"TIME_SHIFT_{time_bin}"

            self._append_token(time_token_str)

            # Handle different hit object types
            if isinstance(hit_object, HitComboOsu):
                self._append_token("NEW_COMBO")
            elif isinstance(hit_object, HitCircleOsu):
                pass
            elif isinstance(hit_object, SliderOsu):
                pass
            elif isinstance(hit_object, SpinnerOsu):
                pass
            else:
                print(f"Unknown hit object type: {type(hit_object)}")
                continue


            # Update the last event time
            last_event_time_ms = hit_object.start_time

            