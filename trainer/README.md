# 🎙️ Voice Detection Model Trainer

This sub-project is dedicated to fine-tuning a custom AI Voice Detection model tailored to your specific audio samples and languages (Tamil, English, Hindi, Malayalam, Telugu).

## 🏗️ Architecture
- **Base Model**: `facebook/wav2vec2-large-xlsr-53` (Multilingual)
- **Task**: Audio Classification (Binary: HUMAN vs AI_GENERATED)

## 📁 Directory Structure
- `data/`: Put your training audio files here.
    - `real/`: Human voice samples.
    - `fake/`: AI generated voice samples.
- `output/`: Fine-tuned model checkpoints will be saved here.
- `train.py`: Main fine-tuning script.
- `prepare_data.py`: Script to convert audio folders into Hugging Face datasets.

## 🚀 Getting Started
1. **Collect Data**: The more data you have, the better the accuracy. Aim for at least 100-500 samples per category per language.
2. **Setup Environment**:
   ```bash
   pip install transformers datasets torch torchaudio accelerate
   ```
3. **Run Training**:
   ```bash
   python train.py
   ```

## 🔧 Why a Custom Model?
The public models (`mo-thecreator`, etc.) are trained on general datasets. A custom model fine-tuned on **your specific AI voices** (e.g., from specific TTS engines you use) will have much higher accuracy for your use case.
