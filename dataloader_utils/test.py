from osu_to_token import OsuToToken
from vocabulary import BeatmapVocabulary
from quantizers import Quantizers
from token_to_osu import TokenToOsu


vocab = BeatmapVocabulary(
    time_shift_bins=64, #128
    coord_x_bins=128, #32
    coord_y_bins=96, #24
    max_slider_repeats=4,
    spinner_duration_bins=16,
    beat_length_bins=64,
    slider_velocity_bins=32,
    supported_time_signatures=[3, 4, 5, 6, 7],
    slider_max_relative_delta=128
)

quantizers = Quantizers(vocab)


osu_tokens = OsuToToken(
    osu_file_path="beatmapparser/harmony/harmony-hard.osu",
    quantizers=quantizers, 
    vocab=vocab,
    max_seq_len=8192,
    node_parser_script='/Users/sean/Documents/deep-osu/parser/parse_osu_gemini_2.js'
)

#print("Token IDs:", osu_tokens.token_id_sequence)
for i, token_id in enumerate(osu_tokens.token_id_sequence):
    token_str = vocab.get_token(token_id)
    print(f"Token ID {i}: {token_id} -> Token String: {token_str}")
    """
    token = vocab.get_token(token_id)
    if "COORD" in token:
        if "X" in token:
            print("X Coordinate:", quantizers.dequantize_coord(int(token.split("_")[2], 16), axis="x"))
        elif "Y" in token:
            print("Y Coordinate:", quantizers.dequantize_coord(int(token.split("_")[2], 16), axis="y"))

    """
print("Token IDs Length:", len(osu_tokens.token_id_sequence))
print("Vocabulary Size:", vocab.vocab_size)


token_to_osu = TokenToOsu(
    quantizers=quantizers,
    vocab=vocab,
    node_encoder_script='/Users/sean/Documents/deep-osu/parser/encode_osu.js'
)

osu_str = token_to_osu.get_osu_string(
    token_id_sequence=osu_tokens.token_id_sequence,
    metadata_overrides={
        'title': 'Test Title',
        'artist': 'Test Artist',
        'creator': 'Test Creator',
        'version': 'Test Version',
        'source': 'Test Source',
        'tags': 'Test Tag',
        'beatmap_id': 123456,
        'beatmap_set_id': 654321,
        'difficulty_rating': 5.0,
        'max_combo': 1000,
        'drain_time': 300000,
        'hp_drain_rate': 5.0,
        'circle_size': 4.0,
        'overall_difficulty': 5.0,
        'approach_rate': 7.0,
        'slider_multiplier': 1.4,
        'slider_tick_rate': 1.0,
    }
)

print("Generated osu! string:")
print(osu_str)
# Save the generated osu! string to a file
with open("/Users/sean/Documents/deep-osu/dataloader_utils/beatmapparser/harmony/harmony-hard_converted.osu", "w") as f:
    f.write(osu_str)
