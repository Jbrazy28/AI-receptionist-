!pip install -U openai-whisper
import whisper

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
import requests
import json
def convert_text_to_speech(text):
    url = 'https://api.openai.com/v1/engines/whisper-3.5.0/completions'
    data = {
        'prompt': text,
        'max_tokens': 100,
        'temperature': 0.6
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_API_KEY'
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_json = response.json()
    if 'choices' in response_json:
        audio = response_json['choices'][0]['audio']
        return audio
    return None
# Example usage
text = 'Hello, how are you?'
audio_data = convert_text_to_speech(text)
# Save audio to a file
if audio_data:
    with open('output.wav', 'wb') as f:
        f.write(audio_data)
        print('Speech audio saved to output.wav')
else:
    print('Text to speech conversion failed.')
