This document provides a more in-depth, step-by-step explanation of the plan to create an autoregressive model for generating .osu beatmaps using a CNN Encoder and Transformer Decoder architecture.
Phase 1: Data Representation - Defining the Language
This phase is foundational. We are creating the vocabulary and grammar the model will use to read and write beatmaps. The choices made here directly impact the model's ability to learn and the quality of the generated output.
* Step 1.1: Define the Core Information to Capture
   * Explanation: To represent a beatmap faithfully, we must capture all elements that define gameplay and timing.
      * Map Boundaries (SOS, EOS): Signal the beginning and end of the beatmap sequence for the model.
      * Time Progression (TIME_SHIFT): Crucial for synchronizing objects with the music. We need to represent the time difference between consecutive events (hit objects).
      * Object Types (PLACE_CIRCLE, START_SLIDER, PLACE_SPINNER): Define the fundamental actions available to the player (click, hold/drag, spin).
      * Object Positions (COORD_X, COORD_Y): Determine where on the screen the player interacts.
      * Slider Paths (START_SLIDER(type), ADD_SLIDER_ANCHOR, END_SLIDER(repeats)): Sliders require defining their shape (Linear, Bezier, Perfect circle), the control points that define the curve, and how many times the player traverses the path back and forth.
      * Spinner Durations (END_SPINNER(duration)): Spinners require knowing how long the player needs to spin.
      * Combo Management (NEW_COMBO): Affects scoring and visual grouping of objects. It's a flag applied before an object is placed.
* Step 1.2: Design the Token Vocabulary
   * Decision Rationale: We use quantized parameters and separate action/parameter tokens.
      * Quantization: Converts continuous values (time, position, duration) into discrete bins. This turns the prediction problem into a classification task (predicting the correct bin/token) which is generally easier for sequence models than directly regressing continuous values. It also keeps the vocabulary size manageable.
      * Separation: Using tokens like PLACE_CIRCLE followed by COORD_X(15) and COORD_Y(22) instead of a single combined token CIRCLE_X15_Y22 drastically reduces vocabulary size (e.g., 1 + 32 + 32 tokens vs. 1 * 32 * 32 tokens). It also encourages the model to learn the compositional structure (an action followed by its parameters). The trade-off is potentially longer sequences.
   * Vocabulary Categories (Detailed):
      * Special Tokens:
         * <PAD> (ID 0): Used to fill shorter sequences in a batch to match the length of the longest sequence. The model learns to ignore this token.
         * <SOS> (ID 1): Start Of Sequence. Prepended to every sequence to signal the beginning of generation/processing.
         * <EOS> (ID 2): End Of Sequence. Appended to signal the end of the beatmap content. The model learns to predict this token when the map is complete.
      * Time Tokens (TIME_SHIFT(bin)):
         * Represents the time elapsed since the previous event.
         * Quantization: Logarithmic scaling is preferred. Small time differences (e.g., 1/4 vs 1/3 beat) are perceptually more distinct and rhythmically important than the same absolute difference at longer intervals. We divide the range of possible time differences (e.g., 1ms to ~5000ms) into bins where bin widths increase logarithmically.
         * Example (64 bins): TIME_SHIFT_0 might cover 0-10ms, TIME_SHIFT_1 11-25ms, ..., TIME_SHIFT_63 > 2000ms. The exact boundaries depend on the chosen log base and range.
         * Implementation: A function quantize_time_shift(dt_ms) maps milliseconds dt_ms to a bin index (0-63). A corresponding dequantize_time_shift(bin) function maps the bin index back to an approximate dt_ms (e.g., the midpoint or start of the bin's range).
      * Object Type & Action Tokens:
         * NEW_COMBO: A flag token. When encountered, the next Circle, Slider, or Spinner placed will start a new color combo.
         * PLACE_CIRCLE: Signals that the next COORD_X and COORD_Y tokens define a hit circle at the current time.
         * START_SLIDER_L, START_SLIDER_B, START_SLIDER_P: Signal the beginning of a Linear, Bezier, or Perfect Circle slider, respectively. The next COORD_X/COORD_Y define the start position.
         * ADD_SLIDER_ANCHOR: Signals that the next COORD_X/COORD_Y define an anchor point (for Bezier/Perfect) or intermediate point (for Linear, though less common). Crucially, these coordinates should likely represent the relative offset (dx, dy) from the previous slider point, not absolute coordinates, to make learning paths easier. This requires a separate quantization scheme for relative coordinates.
         * END_SLIDER_0R...END_SLIDER_4R: Signal the end of the slider definition. The number indicates repeats (0 = 1 pass, 1 = 2 passes, etc.). Limiting repeats (e.g., to 4) caps vocabulary size; higher repeats are rare. The final COORD_X/COORD_Y before this token define the slider's end point (or the last anchor).
         * PLACE_SPINNER: Signals a spinner at the current time (position is fixed at center screen 256,192).
         * END_SPINNER_DUR0...END_SPINNER_DUR15: Signals the end of the spinner and provides its duration via a quantized bin. Linear quantization might suffice here (e.g., 0-500ms, 501-1000ms, ...).
      * Coordinate Tokens (COORD_X(bin), COORD_Y(bin)):
         * Represents absolute position for object starts, or relative position (dx, dy) for slider anchors.
         * Normalization: Raw pixel coordinates (X: 0-512, Y: 0-384) are normalized to a 0.0-1.0 range by dividing by the playfield dimensions. This makes the values independent of the specific resolution.
         * Quantization: Linear quantization is suitable here. Divide the 0-1 range into bins.
         * Example (32 bins): COORD_X_0 (0.0-0.03125), ..., COORD_X_31 (0.96875-1.0). Similarly for Y. This gives 32 X-tokens and 32 Y-tokens (total 64 coordinate tokens).
         * Precision: 512px / 32 bins = 16px precision on X. 384px / 32 bins = 12px precision on Y. This is a reasonable starting point. Could be increased (e.g., 64 bins) if needed, at the cost of vocabulary size.
         * Relative Coordinates: For ADD_SLIDER_ANCHOR, the COORD_X/COORD_Y tokens should represent quantized delta values (dx, dy). This requires a different quantization range (e.g., -128px to +128px normalized and then quantized) and potentially separate tokens, or reusing the absolute bins with a defined mapping. Reusing bins is simpler for vocabulary size.
   * Revised Vocabulary Size Estimate: 3 (Special) + 64 (Time) + 1 (NewCombo) + 1 (Circle) + 3 (SliderStart) + 1 (SliderAnchor) + 5 (SliderEnd) + 1 (Spinner) + 16 (SpinnerEnd) + 32 (CoordX) + 32 (CoordY) = 160 tokens. (Assuming coordinate bins are reused for relative deltas).
* Step 1.3: Implement Tokenizer (.osu -> Token Sequence)
   * Input: Filepath to an .osu file.
   * Output: A Python list or PyTorch tensor of integer token IDs.
   * Detailed Process:
      1. Parse: Use a robust .osu parser (like the beatmapparser provided or a similar library) to load hit objects, timing points, and difficulty settings. Handle potential parsing errors gracefully.
      2. Sort: Ensure hit objects are strictly sorted by start_time.
      3. Initialize: token_sequence = [vocab.sos_id]. last_event_time = 0.
      4. Iterate Objects: For each hit_object in sorted list:
         * Time Shift: dt = hit_object.start_time - last_event_time. dt = max(0, dt). time_bin = quantize_time_shift(dt). token_sequence.append(vocab.get_id(f"TIME_SHIFT_{time_bin}")).
         * New Combo: If hit_object has the "new combo" flag set, token_sequence.append(vocab.new_combo_id).
         * Object Specific:
            * Circle: Append PLACE_CIRCLE ID. Quantize hit_object.position.x, hit_object.position.y (normalized) into x_bin, y_bin. Append COORD_X_{x_bin} ID, COORD_Y_{y_bin} ID.
            * Slider: Append START_SLIDER_{type} ID (where type is L, B, or P). Quantize start position x_bin, y_bin. Append COORD_X_{x_bin} ID, COORD_Y_{y_bin} ID. Iterate through hit_object.curve_points: For each anchor_point: Append ADD_SLIDER_ANCHOR ID. Calculate relative delta dx = anchor_point.x - previous_point.x, dy = anchor_point.y - previous_point.y. Quantize dx, dy using the relative coordinate scheme into dx_bin, dy_bin. Append COORD_X_{dx_bin} ID, COORD_Y_{dy_bin} ID. Update previous_point. After loop, quantize hit_object.repeat_count (clamped) into repeat_bin. Append END_SLIDER_{repeat_bin}R ID.
            * Spinner: Append PLACE_SPINNER ID. Calculate duration = hit_object.end_time - hit_object.start_time. Quantize duration into duration_bin. Append END_SPINNER_DUR{duration_bin} ID.
         * Update Time: last_event_time = hit_object.start_time (or hit_object.end_time? Needs careful consideration - likely start_time is correct for sequencing).
      5. Finalize: Append vocab.eos_id.
      6. Return: token_sequence.
* Step 1.4: Implement Detokenizer (Token Sequence -> .osu)
   * Input: A list/tensor of predicted token IDs.
   * Output: A string containing the content for a valid .osu file.
   * Detailed Process:
      1. Initialize: hit_objects = []. current_time = 0. current_x = 0, current_y = 0. new_combo_flag = False. slider_state = None (e.g., dictionary to hold type, start_time, points).
      2. Iterate Tokens: For token_id in sequence (ignore PAD, stop at EOS):
         * Decode token_str = vocab.get_token(token_id).
         * Handle Time: If token_str.startswith("TIME_SHIFT_"): bin = int(token_str.split('_')[-1]). dt = dequantize_time_shift(bin). current_time += dt.
         * Handle Combo: If token_str == "NEW_COMBO": new_combo_flag = True.
         * Handle Actions:
            * If token_str == "PLACE_CIRCLE": object_type_pending = "circle".
            * If token_str.startswith("START_SLIDER_"): slider_state = {'type': token_str[-1], 'start_time': current_time, 'points': []}. object_type_pending = "slider_start".
            * If token_str == "ADD_SLIDER_ANCHOR": object_type_pending = "slider_anchor".
            * If token_str.startswith("END_SLIDER_"): If slider_state: Decode repeats from token. Create SliderOsu object using slider_state data (calculate length based on points/timing if needed, or leave approximate). Add to hit_objects. Reset slider_state = None, new_combo_flag = False.
            * If token_str == "PLACE_SPINNER": spinner_start_time = current_time. object_type_pending = "spinner".
            * If token_str.startswith("END_SPINNER_DUR"): If object_type_pending == "spinner": Decode duration_bin to duration. Create SpinnerOsu object (start_time=spinner_start_time, end_time=spinner_start_time + duration). Add to hit_objects. Reset new_combo_flag = False, object_type_pending = None.
         * Handle Coordinates:
            * If token_str.startswith("COORD_X_"): x_bin = int(token_str.split('_')[-1]). Store x_bin.
            * If token_str.startswith("COORD_Y_"): y_bin = int(token_str.split('_')[-1]). Store y_bin. Now check object_type_pending:
               * If "circle": Dequantize x_bin, y_bin to x, y. Create HitCircleOsu at current_time, position=(x, y), apply new_combo_flag. Add to hit_objects. Reset flags/pending state.
               * If "slider_start": Dequantize x_bin, y_bin to x, y. slider_state['points'].append((x, y)). Reset pending state.
               * If "slider_anchor": Dequantize x_bin, y_bin using relative dequantizer to get dx, dy. Calculate new_x = last_slider_point.x + dx, new_y = last_slider_point.y + dy. slider_state['points'].append((new_x, new_y)). Reset pending state.
      3. Format Output: Create the full .osu file string. Include standard headers ([General], [Metadata], [Difficulty] - potentially using default values or values passed as arguments), [Events], [TimingPoints] (a simple default like 120 BPM might suffice initially), and finally the generated [HitObjects] section formatted correctly.
      4. Return: .osu string.
Phase 2: Data Loading and Preprocessing
This phase connects the raw data (audio, .osu files) to the model, preparing it in the batched tensor format required for training.
* Step 2.1: Update Dataset and DataLoader
   * BeatmapDataset (torch.utils.data.Dataset):
      * __init__: Takes a list of beatmap identifiers (e.g., file paths or IDs), paths to audio files, max_audio_len, max_target_seq_len, tokenizer instance, audio feature extraction parameters.
      * __len__: Returns the total number of beatmaps in the dataset.
      * __getitem__(idx): Defines how to load and process a single sample:
         1. Load Audio: Use torchaudio.load or librosa.load to get waveform, sample_rate. Resample if necessary to a consistent rate.
         2. Extract Features: Calculate Mel Spectrogram using torchaudio.transforms.MelSpectrogram or librosa.feature.melspectrogram. Parameters: n_mels (e.g., 80), n_fft (e.g., 1024), hop_length (determines time resolution, e.g., 256). Convert power spectrogram to dB scale. Result features shape: [n_mels, audio_frames]. Transpose to [audio_frames, n_mels].
         3. Parse .osu: Use the parser (e.g., BeatmapOsu(osu_filepath)) to get beatmap object.
         4. Extract Difficulty: Get AR, CS, OD, HP from the parsed object. Create difficulty_tensor = torch.tensor([ar, cs, od, hp], dtype=torch.float32).
         5. Tokenize: target_token_sequence = self.tokenizer.tokenize(parsed_osu). This returns a list of token IDs (including SOS/EOS).
         6. Pad/Truncate:
            * Audio: If features.shape[0] > max_audio_len, truncate. If < max_audio_len, pad along the time dimension (e.g., with zeros). Create an audio_padding_mask (True where padded).
            * Target Tokens: If len(target_token_sequence) > max_target_seq_len, truncate (ensure EOS is kept if possible, or replace last token with EOS). If < max_target_seq_len, pad with PAD_ID. Create a target_padding_mask (True where padded).
         7. Return: (padded_features, audio_padding_mask, difficulty_tensor, padded_target_tokens, target_padding_mask).
   * DataLoader (torch.utils.data.DataLoader):
      * Wraps the BeatmapDataset.
      * Handles batching: Groups multiple samples returned by __getitem__ into a single batch. Tensors will have an added batch dimension (e.g., [batch_size, max_audio_len, n_mels]).
      * Handles shuffling: Randomizes the order of samples each epoch.
      * collate_fn: The default collate function usually works correctly if __getitem__ returns tensors of consistent sizes (due to padding). It stacks the corresponding elements from each sample into batch tensors.
Phase 3: Model Architecture (CNN Encoder + Transformer Decoder)
This phase defines the neural network components that learn the mapping from audio and difficulty to the beatmap token sequence.
* Step 3.1: CNN Encoder
   * Goal: To learn relevant local patterns (like rhythm, intensity changes) in the audio spectrogram and reduce the sequence length, making the subsequent Transformer computationally cheaper.
   * Architecture:
      * Input: features [batch, max_audio_len, n_mels].
      * Layers: A stack of 1D convolutional layers (nn.Conv1d) is suitable, operating along the time dimension.
         * Example:
# Input: [B, T_audio, n_mels] -> Permute: [B, n_mels, T_audio]
nn.Conv1d(in_channels=n_mels, out_channels=256, kernel_size=5, stride=2, padding=2) # Downsample x2
nn.ReLU()
nn.LayerNorm(...) # Normalize across channel dim
nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2) # Downsample x4
nn.ReLU()
nn.LayerNorm(...)
nn.Conv1d(in_channels=512, out_channels=d_model, kernel_size=5, stride=2, padding=2) # Downsample x8
# Output: [B, d_model, T_audio / 8] -> Permute: [B, T_audio / 8, d_model]

   * Key Choices: kernel_size (e.g., 3 or 5) defines the local time window. stride > 1 performs downsampling. padding maintains size or controls output size. Activation (ReLU/GELU) adds non-linearity. Normalization (LayerNorm/BatchNorm) stabilizes training.
   * Final Projection: Ensure the output channel dimension matches the Transformer's d_model.
   * Output: audio_memory [batch, reduced_audio_len, d_model]. Also return the memory_key_padding_mask corresponding to the padded regions in the reduced length sequence.
   * Step 3.2: Difficulty Embedding
   * Goal: Convert the low-dimensional difficulty vector into the high-dimensional d_model space.
   * Architecture:
   * Input: difficulty_tensor [batch, num_difficulty_features] (e.g., 4 features).
   * Layer: A single nn.Linear(num_difficulty_features, d_model).
   * Output: difficulty_embedding [batch, d_model].
   * Step 3.3: Token Embedding
   * Goal: Convert integer token IDs into dense vector representations.
   * Architecture:
   * Input: target_token_sequence [batch, max_target_seq_len].
   * Layer: nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID). The padding_idx tells the layer to output a zero vector (or a non-learned vector) for PAD tokens and not update its gradient.
   * Output: token_embeddings [batch, max_target_seq_len, d_model].
   * Step 3.4: Positional Encoding
   * Goal: Inject information about the absolute or relative position of each token, as the Transformer itself is permutation-invariant.
   * Architecture: Standard sinusoidal positional encoding is common. It uses fixed sine and cosine functions of different frequencies. Alternatively, use a learned nn.Embedding based on position indices.
   * Application: Add the positional encoding element-wise to the token_embeddings. final_embeddings = token_embeddings + positional_encoding.
   * Step 3.5: Transformer Decoder
   * Goal: Generate the output sequence token by token, using information from the audio (memory) and the previously generated tokens (tgt).
   * Architecture: nn.TransformerDecoder which stacks multiple nn.TransformerDecoderLayers. Each layer contains:
   1. Self-Attention: Attends to the previously generated tokens (masked to prevent seeing future tokens). Input: tgt.
   2. Cross-Attention: Attends to the output of the CNN encoder (memory). Input: output of self-attention, memory.
   3. Feed-Forward Network: Processes the output of the cross-attention.
   * Inputs:
   * tgt: The sequence being generated (during training, the ground truth shifted right; during inference, the sequence generated so far). Shape: [batch, current_seq_len, d_model]. This includes token embeddings, difficulty embedding, and positional encoding.
   * memory: Output from the CNN Encoder. Shape: [batch, reduced_audio_len, d_model].
   * tgt_mask: Square causal mask [current_seq_len, current_seq_len] generated by nn.Transformer.generate_square_subsequent_mask(). Ensures a position i can only attend to positions <= i.
   * memory_mask: Optional, usually not needed if padding is handled by memory_key_padding_mask.
   * tgt_key_padding_mask: Boolean tensor [batch, current_seq_len]. True for positions that are PAD tokens in the tgt sequence. Prevents attention calculation on padding.
   * memory_key_padding_mask: Boolean tensor [batch, reduced_audio_len]. True for positions that correspond to padding in the original audio after downsampling by the CNN.
   * Parameters: d_model (internal dimension, e.g., 512), nhead (number of attention heads, e.g., 8, must divide d_model), num_decoder_layers (e.g., 6), dim_feedforward (hidden size of FFN, e.g., 2048), dropout (e.g., 0.1).
   * Output: decoder_output [batch, current_seq_len, d_model].
   * Step 3.6: Output Head
   * Goal: Project the final decoder hidden states back into the vocabulary space to get scores (logits) for each possible next token.
   * Architecture:
   * Input: decoder_output [batch, max_target_seq_len, d_model].
   * Layer: A single nn.Linear(d_model, vocab_size).
   * Output: output_logits [batch, max_target_seq_len, vocab_size].
   * Step 3.7: Combining Difficulty (Refined)
   * Method: Add the difficulty_embedding [batch, d_model] to the token_embeddings [batch, max_target_seq_len, d_model]. The difficulty embedding needs to be unsqueezed and expanded (difficulty_embedding.unsqueeze(1).expand(-1, max_target_seq_len, -1)) to match the sequence length dimension before adding.
   * Placement: Add it after token embedding but before adding positional encoding. combined_embeddings = token_embeddings + expanded_difficulty_embedding. final_input_embeddings = combined_embeddings + positional_encoding.
   * Justification: This injects the difficulty context into the representation of every token input to the decoder, influencing all subsequent calculations and predictions.
Phase 4: Training
This phase focuses on optimizing the model's parameters using the prepared data and the defined architecture.
   * Step 4.1: Loss Function
   * Choice: nn.CrossEntropyLoss.
   * Explanation: This loss function is standard for multi-class classification. It internally applies a LogSoftmax to the model's output logits and then calculates the Negative Log Likelihood Loss (NLLLoss) against the target class indices. It expects raw logits as input.
   * Usage:
   * criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID): The ignore_index ensures that loss is not calculated for positions where the target label is PAD_ID.
   * Input shapes: The logits need to be [batch * seq_len, vocab_size] and target labels [batch * seq_len]. So, reshape: loss = criterion(output_logits.view(-1, vocab_size), target_labels.view(-1)).
   * Step 4.2: Optimizer
   * Choice: torch.optim.AdamW.
   * Explanation: AdamW modifies the standard Adam optimizer by decoupling weight decay from the gradient update, which often leads to better generalization in models like Transformers.
   * Usage: optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01). Learning rate (lr) and weight_decay are key hyperparameters to tune.
   * Step 4.3: Learning Rate Scheduler
   * Purpose: Gradually decrease the learning rate during training to help convergence and avoid overshooting the optimal parameters.
   * Options:
   * ReduceLROnPlateau: Monitors a metric (e.g., validation loss). Reduces LR by a factor if the metric doesn't improve for patience epochs. scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5). Step using scheduler.step(validation_loss).
   * Warmup + Decay (e.g., Cosine): Increase LR linearly for a fixed number of warmup_steps, then decay it following a cosine curve. Often implemented manually or using libraries like transformers.
   * Justification: Essential for stable training of large models.
   * Step 4.4: Training Loop (Pseudocode)
model.train()
for epoch in range(num_epochs):
   for batch in train_dataloader:
       # 1. Get data and move to device
       features, audio_mask, difficulty, targets, target_mask = batch
       features, audio_mask, difficulty, targets, target_mask = features.to(device), audio_mask.to(device), difficulty.to(device), targets.to(device), target_mask.to(device)

       # 2. Prepare decoder inputs/labels
       # <SOS> token id = 1
       decoder_input_tokens = targets[:, :-1] # All but last token
       target_labels = targets[:, 1:] # All but <SOS> token

       # 3. Generate masks for Transformer Decoder
       tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input_tokens.size(1)).to(device)
       # Use target_mask derived from dataloader for tgt_key_padding_mask (adjusting for shift)
       tgt_padding_mask = (decoder_input_tokens == PAD_ID) # Create padding mask for decoder input

       # 4. Zero gradients
       optimizer.zero_grad()

       # 5. Forward Pass
       audio_memory, memory_padding_mask = model.encode(features, audio_mask) # Pass audio mask to encoder
       output_logits = model.decode(decoder_input_tokens, audio_memory, difficulty, # Pass difficulty
                                    tgt_causal_mask, tgt_padding_mask, memory_padding_mask)

       # 6. Calculate Loss (ignore padding in labels)
       loss = criterion(output_logits.reshape(-1, vocab_size), target_labels.reshape(-1))

       # 7. Backward Pass
       loss.backward()

       # 8. Gradient Clipping
       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

       # 9. Optimizer Step
       optimizer.step()

       # 10. Scheduler Step (if per-step scheduler)
       # scheduler.step()

   # --- End of Epoch ---
   # 11. Run Validation Loop
   validation_loss = evaluate(model, val_dataloader, criterion, device)
   print(f"Epoch {epoch}, Loss: {loss.item()}, Val Loss: {validation_loss}")

   # 12. Scheduler Step (if per-epoch or validation-based scheduler)
   # scheduler.step(validation_loss)

Phase 5: Inference (Generation)
This phase uses the trained model to create new beatmaps.
      * Step 5.1: Setup
      1. model.eval(): Set model to evaluation mode (disables dropout, etc.).
      2. with torch.no_grad():: Disable gradient calculations to save memory and computation.
      3. Pre-compute:
      * Load and process the input audio -> features.
      * Pass features through CNN Encoder -> audio_memory, memory_padding_mask. This is done only once per generation.
      * Embed the target difficulty_tensor -> difficulty_embedding.
      * Step 5.2: Autoregressive Generation Loop
      1. Initialize: generated_tokens = torch.tensor([[SOS_ID]], device=device). Start with the SOS token in a batch of size 1.
      2. Loop: for _ in range(max_target_seq_len):
      * Get current sequence: current_input_tokens = generated_tokens.
      * Prepare masks for the current length: tgt_causal_mask, tgt_padding_mask (no padding initially).
      * Forward Pass (Decoder only): Call the model's decode function (or relevant parts) using current_input_tokens, pre-computed audio_memory, difficulty_embedding, and masks.
      * decoder_output = model.decode(...)
      * Get logits for the very last token prediction: next_token_logits = decoder_output[:, -1, :] [1, vocab_size].
      * Sampling Strategy:
      * Greedy: next_token_id = torch.argmax(next_token_logits, dim=-1). Simple, deterministic, but often repetitive.
      * Top-k: Keep k highest probability tokens. Apply softmax to next_token_logits, set probabilities of others to 0, re-normalize, sample using torch.multinomial.
      * Top-p (Nucleus): Recommended. Apply softmax. Sort probabilities descending. Find the smallest set whose cumulative probability >= p. Keep only these tokens, set others to 0, re-normalize, sample using torch.multinomial. Balances quality and diversity.
      * Beam Search: Maintain k parallel candidate sequences (beams). At each step, expand each beam, calculate probabilities, keep the top k overall sequences. More complex, potentially higher quality, less diverse.
      * Sample next_token_id using the chosen strategy.
      * Append: generated_tokens = torch.cat([generated_tokens, next_token_id.unsqueeze(0)], dim=1).
      * Check EOS: If next_token_id.item() == EOS_ID, break the loop.
      * Step 5.3: Detokenization
      1. Get the final sequence from generated_tokens. Convert tensor to list.
      2. Remove SOS and EOS tokens. Handle any potential PAD tokens if generation stopped early.
      3. Pass the cleaned list of token IDs to the Detokenizer function (Step 1.4).
      4. Output: The generated .osu file content as a string.
This detailed plan covers the key considerations and steps for each phase. Remember that implementation will require careful coding, debugging, and hyperparameter tuning.