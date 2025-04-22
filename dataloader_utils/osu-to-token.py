from dataloader_utils.beatmapparser.parser import BeatmapOsu
from dataloader_utils.beatmapparser.enums import HitObjectType
from dataloader_utils.beatmapparser.models import HitComboOsu, HitCircleOsu, SliderOsu, SpinnerOsu

from dataloader_utils.vocabulary import BeatmapVocabulary
from dataloader_utils.quantizers import Quantizers

quantizers = Quantizers(time_shift_bins=64, coord_bins=32, max_slider_repeats=4, spinner_duration_bins=16)
vocab = BeatmapVocabulary(quantizers)
parser = BeatmapOsu("path/to/beatmap.osu")

def tokenize_osu(parsed_osu, vocab, quantizers):
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
        time_token_id = vocab.get_id(time_token_str)

        # handle if token doesn't exist
        if time_token_id is None: 
            print(f"Token not found for time bin: {time_bin}")
            continue
    
        # Append the time token to the sequence
        token_id_sequence.append(time_token_id)

        # Handle different hit object types
        if isinstance(hit_object, HitComboOsu):
            new_combo_id = vocab.get_id("NEW_COMBO")
            if new_combo_id is not None:
                token_id_sequence.append(new_combo_id)
            else:
                print("NEW_COMBO token not found")
                continue
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

        