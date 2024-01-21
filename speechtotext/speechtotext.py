import speech_recognition as sr

def speech(fn):
    r = sr.Recognizer()

    hellow=sr.AudioFile(fn)
    with hellow as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
        print("Text: "+s)
        return s
    except Exception as e:
        print("Exception: "+str(e))