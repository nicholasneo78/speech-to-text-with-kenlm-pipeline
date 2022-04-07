from decoder import *
import utils
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print()

# load the finetuned model and the processor
model = Wav2Vec2ForCTC.from_pretrained('./saved_model/')
processor = Wav2Vec2Processor.from_pretrained('./processor/')
model_ = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
processor_ = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

vocab_dict = processor.tokenizer.get_vocab()
print()
print(vocab_dict['|'])
print()
sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
print()
print(sort_vocab)
vocab = []
for _, token in sort_vocab:
    vocab.append(token)
print()
print(vocab)


# replace the word delimiter with a white space since the white space is used by the decoders
print(processor.tokenizer.word_delimiter_token)
vocab[vocab.index(processor.tokenizer.word_delimiter_token)] = ' '
print(processor.tokenizer.word_delimiter_token)
print()
print(processor.tokenizer.pad_token)
print('-----------------------------------')
print(len(vocab))
print('-----------------------------------')


# define the lm path
lm_path = "lm/4gram_big.arpa.gz" 


# alpha, beta, and beam_wdith SHOULD be tuned on the dev-set to get the best settings
# Feel free to check other inputs of the BeamCTCDecoder
alpha=0
beta=0
beam_width = 1024 # finding all possible combinations of probability

beam_decoder = BeamCTCDecoder(vocab, lm_path=lm_path,
                                 alpha=alpha, beta=beta,
                                 cutoff_top_n=40, cutoff_prob=1.0,
                                 beam_width=beam_width, num_processes=16,
                                 blank_index=vocab.index(processor.tokenizer.pad_token))


greedy_decoder = GreedyDecoder(vocab, blank_index=vocab.index(processor.tokenizer.pad_token))


# load test audio file
audio_files_paths = ['./datasets/magister_data_flac_16000/test/11039/2614000/11039-2614000-0000.flac', './datasets/magister_data_flac_16000/test/11039/2614000/11039-2614000-0001.flac']

print(f'Load audio files: "{audio_files_paths}"')
batch_audio_files, sampling_rate = utils.load_audio_files(audio_files_paths)
print(batch_audio_files)
print(batch_audio_files[0].shape)

print('Get logits from the Wav2Vec2ForCTC model....')
logits, max_signal_length = utils.get_logits(batch_audio_files, model, processor, device)



print()
print()
print('-----------------------------------')
print(logits.shape)
print('-----------------------------------')
print(logits)
print()
print()


print('Decoding using the Greedy Decoder....')
greedy_decoded_output, greedy_decoded_offsets = greedy_decoder.decode(logits)

print('Decoding using the Beam Search Decoder....')
beam_decoded_output, beam_decoded_offsets = beam_decoder.decode(logits)



print('Printing the output of the first audio file...\n')
#print('Greedy Decoding Output:', greedy_decoded_output[0][0]) #initial
print('Greedy Decoding Output:', greedy_decoded_output[0])
print()
print('#'*85)
print()
print('Beam Search Decoding Output:', beam_decoded_output[0]) # print the top prediction of the beam search

print('Compute Segments....')
batch_segments_list_greedy = utils.get_segments(logits, greedy_decoded_output, max_signal_length, sampling_rate, vocab)
batch_segments_list_beam = utils.get_segments(logits, beam_decoded_output, max_signal_length, sampling_rate, vocab)

print('Printing the first segment (word) of the first audio file...')
print()
print('#'*85)
print()
print('Greedy Decoding Output:', batch_segments_list_greedy[0])
print()
print('Beam Search Decoding Output:', batch_segments_list_beam[0])

print('Done!!')