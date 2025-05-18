# VoiceGender — Real-Time Gender & Pitch Detection

## About
VoiceGender is a Python 3 desktop app that:

1. Listens to your microphone in real time.  
2. Predicts speaker gender with a fine-tuned ECAPA-TDNN model (`JaesungHuh/voice-gender-classifier`).  
3. Displays a femininity score (0–100 %) and the 10 most recent predictions.  
4. Plots fundamental frequency (pitch) for the last 15 s.  
5. Transcribes speech to French text via Google Speech Recognition.  
6. Logs everything to `history.txt`.

Runs fully offline except for the optional Google STT request.

## Requirements
* Python ≥ 3.9  
* Working microphone  
* (Optional) CUDA-enabled PyTorch for GPU acceleration  

See `requirements.txt` for exact package versions.

## Installation
```bash
git clone <your-repo>
cd VoiceGender
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt


##TODO

- [ ] Data ANalyse of historique
- [ ] make the capture still workin while analyzing curent audio
- [ ] update UX
