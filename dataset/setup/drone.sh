wget "https://drive.google.com/uc?export=download&id=1AYKP7N8d-RR_hY5jEM3qxF1n0CGsnUkd&confirm=yes" -O "kaggle.json"
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download bulentsiyah/semantic-drone-dataset
unzip semantic-drone-dataset.zip -d data
rm semantic-drone-dataset.zip