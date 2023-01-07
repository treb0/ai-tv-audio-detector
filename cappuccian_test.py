import os
import tensorflow as tf 
import tensorflow_io as tfio
import sys

SAMPLES = os.path.join('audios', 'samples')
SAMPLE_NAME = 'pedidos_ya_te_salvamos_el_verano'
BROADCASTS = os.path.join('audios', 'broadcasts')

#ten_min_length = 9628725
#ten_min_length = 6019376
#ten_min_length = 48000*100*2
ten_min_length = 9857115


def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


## WORKING WITH ONLY 1 SAMPLE / SONG

# Create samples dataset
SAMPLES = os.path.join('audios', 'samples')
samples = tf.data.Dataset.list_files(SAMPLES+'/*.wav')
data = tf.data.Dataset.zip((samples, tf.data.Dataset.from_tensor_slices(tf.ones(len(samples)))))

sample_length = len(load_wav_16k_mono( os.path.join('audios', 'samples',SAMPLE_NAME+'.wav')))

print('')
print('sample length: '+str(sample_length))
print('')

def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    #wav = wav[:48000]
    #zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    #wav = tf.concat([zero_padding, wav],0)
    # quitamos estas tres lineas porque toma una primera parte de todos los samples nada mas, y no aplica en nuestro caso
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label


data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

print('')
print('data length:'+str(len(data)))
print('data:')
print(data)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257,1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

print('')
print('model.summary:')
model.summary()

# 9. Build Forest Parsing Functions

def load_mp3_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    res = tfio.audio.AudioIOTensor(filename)
    # Convert to tensor and combine channels 
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
    # Extract sample rate and cast
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16 kHz
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

ten_min_length = 9857115

while True:

    if ten_min_length >= 9857127: break

    ten_min_length += 1
    
    try:

        print('trying run with ten_min_length = '+str(ten_min_length))


        # 9.2 Build Function to Convert Clips into Windowed Spectrograms

        def preprocess_mp3(sample, index):
            sample = sample[0]
            zero_padding = tf.zeros([ten_min_length] - tf.shape(sample), dtype=tf.float32)
            wav = tf.concat([zero_padding, sample],0)
            spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
            spectrogram = tf.abs(spectrogram)
            spectrogram = tf.expand_dims(spectrogram, axis=2)
            return spectrogram




        mp3 = os.path.join('audios', 'broadcasts', 'tyc.mp3')

        wav = load_mp3_16k_mono(mp3)

        audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav
                                                                ,wav
                                                                ,sequence_length = ten_min_length
                                                                ,sequence_stride = ten_min_length
                                                                ,batch_size=1
                                                                )

        print('')
        print('len(audio_slices)='+str(len(audio_slices)))
        print('')

        print('audio_slices:')
        print(audio_slices)

        #samples, index = audio_slices.as_numpy_iterator().next()
        audio_slices = audio_slices.map(preprocess_mp3)
        audio_slices = audio_slices.batch(64)

        print('')
        print('audio_slices:')
        print(audio_slices)


        print('')
        print('start model prediction')

        #sys.exit()

        yhat = model.predict(audio_slices)
        yhat_simple = [1 if prediction > 0.5 else 0 for prediction in yhat]

        print('')
        print('yhat:')
        print(yhat_simple)

        print('SUCCESS')
        break


    except:


        print('')
        print('failed')
        
