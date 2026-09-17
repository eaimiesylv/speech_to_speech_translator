import whisper

model = whisper.load_model("medium")

result = model.transcribe(
    "ekaro.m4a",
    language="yo",
    task="transcribe",
    fp16=False,
    temperature=0
)

print(result["text"])