import subprocess, pandas as pd
def test_dataset_size():
    df = pd.read_csv('data/retail_sales.csv')
    assert len(df) > 400_000
def test_train_runs(tmp_path):
    out = tmp_path / 'cv.csv'
    subprocess.check_call(['python3','scripts/train_forecasts.py','--output_csv', str(out)])
    assert out.exists()
