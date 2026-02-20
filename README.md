# error-analysis-ui

A lightweight NiceGUI app to browse error samples stored in Parquet files.

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python app.py --parquet data/error_samples.parquet --host 127.0.0.1 --port 8080
```

Open: http://127.0.0.1:8080

## Notes
- The app can read images stored as blobs or via file paths inside the Parquet.
- If images are stored as paths, ensure the files exist at those paths on your machine.
