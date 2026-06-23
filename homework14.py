import gtts, speech_recognition, librosa, soundfile

def synthesize(text, lang, filename):
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)

def make_a_corpus(texts, languages, filenames):
    recognized_texts = []
    recognizer = speech_recognition.Recognizer()

    for text, lang, root in zip(texts, languages, filenames):
        mp3file = root + ".mp3"
        wavfile = root + ".wav"

        synthesize(text, lang, mp3file)

        y, sr = librosa.load(mp3file, sr=None)
        soundfile.write(wavfile, y, sr)

        with speech_recognition.AudioFile(wavfile) as source:
            audio = recognizer.record(source)

        recognized = recognizer.recognize_google(audio, language=lang)
        recognized_texts.append(recognized)

    return recognized_texts