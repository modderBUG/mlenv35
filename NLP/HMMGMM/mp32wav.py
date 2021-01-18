from os import path
from pydub import AudioSegment

# files
src = r"C:\Users\wuxiaowei_a\PycharmProjects\mlenv35\NLP\HMMGMM\bugscaner-tts-auido.mp3"
dst = "test.wav"

# for %i in (C:\Users\wuxiaowei_a\PycharmProjects\mlenv35\NLP\HMMGMM\*.mp3) do ffmpeg -i "%i" -acodec pcm_s16le -ac 1 -ar 44100 "%i~nf.wav"
# convert wav to mp3
sound = AudioSegment.from_mp3(r"C:\Users\wuxiaowei_a\PycharmProjects\mlenv35\NLP\HMMGMM\bugscaner-tts-auido.mp3")
sound.export(dst, format="wav")