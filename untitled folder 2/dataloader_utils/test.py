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
    slider_duration_bins=32
)

quantizers = Quantizers(vocab)


osu_tokens = OsuToToken(
    osu_file_path="beatmapparser/harmony/harmony-hard.osu",
    quantizers=quantizers, 
    vocab=vocab,
    max_seq_len=10
)

#print("Token IDs:", osu_tokens.token_id_sequence)
for i, token_id in enumerate(osu_tokens.token_id_sequence):
    token_str = vocab.get_token(token_id)
    print(f"Token ID {i}: {token_id} -> Token String: {token_str}")
    token = vocab.get_token(token_id)
    if "COORD" in token:
        if "X" in token:
            print("X Coordinate:", quantizers.dequantize_coord(int(token.split("_")[2], 16), axis="x"))
        elif "Y" in token:
            print("Y Coordinate:", quantizers.dequantize_coord(int(token.split("_")[2], 16), axis="y"))
print("Token IDs Length:", len(osu_tokens.token_id_sequence))
print("Vocabulary Size:", vocab.vocab_size)


osu = TokenToOsu(
    quantizers=quantizers,
    vocab=vocab
)

osu.create_osu_file(
    token_ids=osu_tokens.token_id_sequence,
    output_path="beatmapparser/harmony/harmony-hard_converted.osu",
    slider_tick_rate=osu_tokens.parsed_osu.slider_tick_rate,
    slider_multiplier=osu_tokens.parsed_osu.slider_multiplier,
    hp_drain_rate=osu_tokens.parsed_osu.hp,
    circle_size=osu_tokens.parsed_osu.cs,
    overall_difficulty=4.44,
    approach_rate=osu_tokens.parsed_osu.ar
)