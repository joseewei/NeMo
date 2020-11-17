import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd

def extract_labels(data, time):
    labels = []
    for pos in time:
        line =  data[(data["start"] <=pos) & (data["start"] + data["dur"] > pos)]
        if len(line) >= 1:    
            labels.append(1)  # change this to overlap speech later
        else:
            labels.append(0)
    return labels


def play(df, df_one, filepath, offset, duration):

    audio, sample_rate = librosa.load(path=filepath,sr=16000, mono=True, 
                                      offset=offset, duration=duration) # load audio snippet
    dur = librosa.get_duration(audio, sr=16000)

#     if offset > dur:
#         raise ValueError(f"Selected audio snippet ({offset}s, {offset+duration}s) is out of boundary (overall duration is {dur} seconds)! Try smaller offert!")
        
#     if dur < offset + duration:
#         print(f"Selected audio snippet ({offset}s, {offset+duration}s) is out of boundary (overall duration is {dur} seconds)! Truncating")
#         duration = dur - offset
    
    plt.figure(figsize=[20,10])
    num = len(df.columns) + 1
    
    # extract ground truth label
    time = np.arange(offset, offset + duration, 0.01)
    label = extract_labels(df_one, time)
    
    for i in range(0, num-1):
        FRAME_LEN = 0.01
        data = df[int(offset * 100): int((offset + duration)*100)]
        len_pred = len(data)
        ax1 = plt.subplot(num+1,1,i+1)
        ax1.plot(np.arange(audio.size) / 16000, audio, 'gray')
        ax1.set_xlim([0, int(dur)+1]) 
        ax1.tick_params(axis='y', labelcolor= 'b')
        ax1.set_ylabel('Signal')
        ax1.set_ylim([-1,  1])
        
        ax2 = ax1.twinx()
        prob = data[df.columns[i]]
        pred = np.where(prob >= 0.5, 1, 0)
        ax2.plot(np.arange(len_pred) * FRAME_LEN, label , 'r',  label='label')
#         ax2.plot(np.arange(len_pred) * FRAME_LEN, pred , 'b',  label='pred')
        ax2.plot(np.arange(len_pred) * FRAME_LEN, prob ,  'g--', label='speech prob')
        
        ax2.tick_params(axis='y', labelcolor='r')
        legend = ax2.legend(loc='lower right', shadow=True)
        ax2.set_title(f'{df.columns[i]}')
        ax2.set_ylabel('Preds and Probas')
        ax2.set_ylim([-0.1,  1.1])

    ax = plt.subplot(num+1,1,i+2)
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=64, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_rate, fmax=8000)
    ax.set_xlim([0, int(dur)+1]) 
    ax.set_title('Mel-frequency spectrogram')
    ax.grid()
    plt.show()
    
    return ipd.Audio(audio, rate=sample_rate)