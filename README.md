# DiveIntoCode Graduation Assignment

## 1 Setup server environment by Conda

### 1.1. Install Anaconda (or Miniconda) in home directory (ie. `/home/<user>/anaconda3/`)

<https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>

### 1.2. Create conda virtual environment (either way below)

#### Create from conda shared environment file

```
conda env create --file conda.yml
```

#### Create a new environment named **dive_graduation** that contains Python 3.9 and pre-defined packages from `pkgs.txt` and `requirements.txt

```
conda create --name dive_graduation python=3.9 --yes --file pkgs.txt
conda activate dive_graduation
pip install -r requirements.txts
```

### 1.3. Activate

#### To use environment

```
conda activate dive_graduation
```

#### To deactivate current running Conda environment

```
conda deactivate
```

#### To update `conda.yml` in case of environment modification (add/update packages, change python version, etc.)

```
conda env export --name dive_graduation > conda.yml
```

<br />

## 2. Data preparation

Run the following scripts in order to preprocess data

```
python preprocess/preocess_nodes.py
python preprocess/preocess_segments_streets.py

# Run preprocess_base_status
python preprocess/preprocess_base_status/process_1.py
python preprocess/preprocess_base_status/process_2.py

# Run preprocess_segment_status
python preprocess/preprocess_segment_status/process_1.py
python preprocess/preprocess_segment_status/process_2.py
python preprocess/preprocess_segment_status/process_3.py
python preprocess/preprocess_segment_status/process_4.py

# Run preprocess_segment_status
python preprocess/preprocess_segment_report/process_1.py
python preprocess/preprocess_segment_report/process_2.py
python preprocess/preprocess_segment_report/process_3.py
```
After preprocessing datasets, zip and upload to Google Drive; then train the model

## 3. Train and evaluate model (Google Colab)

<https://github.com/thanhnguyen2612/diveintocode-ml/blob/master/ITS_DL_bigdata.ipynb>

After training model, download the model and encoders into src.

Structure of repository:
|Path|Description|
|---|---|
| data_origin | data dumped directly from database (MongoDB) |
| dataset/ | folder contains preprocessed data in **Data preparation** |
| dataset/segments.csv | combined information related to segments (id, length, geolocation, street properties, etc.) |
| preprocess/ | scripts to preprocessing original data to useful data for train/test |
| src/ | main source code to serve the model |
| src/encoders/ | encoders exported after training (*.pickle) |
| src/model/ | model exported after training and evaluating best model |
| src/model.py | define neural network model for serving prediction |
| src/interface.py | interface to use model |
| src/utils.py | helper functions |
| main.py | flask app runner |

## 4. Run flask APIs
Start Flask app (default at **localhost:8000**)
```
python main.py
```

Lets test the API with **curl**
### 4.1. Inference
```
curl -d '{"segment_ids":[1, 2, 3],"timestamp":1645671708000}' -H 'Content-Type: application/json' http://localhost:8000/inference
curl -d '{"segment_ids":[1, 2, 3],"timestamp":[1645671708000, 1645674288000]}' -H 'Content-Type: application/json' http://localhost:8000/inference
```

### 4.2. Sequence Inference
```
curl -d '{"segment_ids":[1, 2, 3, 4, 5, 6],"timestamp":1645671708000}' -H 'Content-Type: application/json' http://localhost:8000/seq_inference
```
