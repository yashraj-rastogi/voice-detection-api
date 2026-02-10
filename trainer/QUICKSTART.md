# 🎯 Quick Start Guide: Custom Model Training

## ✅ What's Ready
Your training environment is set up in `x:\voice-detection-api\trainer\` with:
- **Training scripts**: `prepare_data.py`, `train.py`
- **Sample data**: 2 AI samples, 1 human sample (in `data/ai/` and `data/human/`)

## ⚠️ Important: You Need More Data
**Current samples (3 total) are NOT enough for production training.**

### Recommended Data Collection:
- **Minimum**: 100 samples per class (100 human + 100 AI)
- **Better**: 500+ samples per class
- **Best**: 1000+ diverse samples covering:
  - All target languages (Tamil, Hindi, Telugu, Malayalam, English)
  - Different speakers, accents, recording conditions
  - Various AI voice generators (if testing multiple TTS engines)

## 🚀 Training Steps

### Step 1: Collect More Audio Samples
Add audio files to:
- `trainer/data/human/` - Real human recordings
- `trainer/data/ai/` - AI-generated audio

Supported formats: `.wav`, `.mp3`, `.flac`

### Step 2: Prepare Dataset
```powershell
cd x:\voice-detection-api\trainer
python prepare_data.py --data_dir data
```
This creates `train_metadata.csv` and `val_metadata.csv`.

### Step 3: Install Dependencies
```powershell
pip install -r requirements_train.txt
```

### Step 4: Train the Model
```powershell
python train.py
```

**Training time**: 
- With GPU (Google Colab): ~30-60 minutes for 100 samples
- Without GPU (CPU only): Several hours (not recommended)

### Step 5: Use Your Custom Model
After training completes, your model will be saved in `trainer/custom_voice_detector/`.

To use it in your API, update `app/config.py`:
```python
MODEL_NAME = "x:/voice-detection-api/trainer/custom_voice_detector"
```

## 💡 Pro Tips

### Using Google Colab (Recommended)
If you don't have a local GPU:
1. Upload your `data/` folder to Google Drive
2. Open a new Colab notebook
3. Mount Google Drive and run the training scripts there
4. Download the trained model back to your local machine

### Data Quality Matters
- Use high-quality recordings (16kHz or higher)
- Ensure balanced classes (similar number of human vs AI samples)
- Include diverse samples to improve generalization

## 📊 Expected Results
With proper training data:
- **Accuracy**: 85-95% on validation set
- **Inference time**: ~100-300ms per audio clip
- **Model size**: ~300-400MB

## 🆘 Need Help?
Common issues:
- **"Not enough data"**: You need at least 20 samples per class to start
- **"Out of memory"**: Reduce `BATCH_SIZE` in `train.py` (try 4 or 2)
- **"Training too slow"**: Use Google Colab with GPU runtime
