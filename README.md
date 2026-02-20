### EduRobot ER

- ER will listen to the command to activate 
- ER will remember what you said to him in the past (RAG)
- ER will run on low spec machine 
- ER will answer on the speaker with human like voice that will impersonate emotions.

It uses [https://github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) for faster voice to speach synthesis
It uses OpenAI API to answer questions (but can be plugged in to your local LLM server)
It uses LocatTTS engine to read text replies generated to the user over the microphone.



Requirements: 
- Python 3 installed
- access to microphone


for Windows machine:

```bash
winget install Git.Git
winget install Python.Python.3.11
winget install Gyan.FFmpeg
```

Python libraries:
```python
pip install --upgrade openai
pip install faster-whisper
pip install sounddevice
pip install scipy
pip install numpy
pip install pyttsx3
```


How to install: 

