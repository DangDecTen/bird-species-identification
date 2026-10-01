# Bird Species Identifier

Protect the ecosystem? Like the work of BirdCLEF that support more reliable biodiversity monitoring in some specific ecosystems. We aims to build a bird identifier with evaluation inspired by these needs. From simple system to product that really capture the need of protecting the ecosystem.



## Setup and Installation

Running it in Colab

```bash
!git clone https://github.com/<you>/bird-classifier && cd bird-classifier
!pip install -q split-folders
# copy data from Drive to local disk first (reading from Drive during training is slow)
!cp -r /content/drive/MyDrive/bird-data/raw data/raw
# run notebooks/01_data_exploration.ipynb, then:
!python -m src.train    --run-name b0_mnv3small_frozen
!python -m src.evaluate --run-name b0_mnv3small_frozen
!python -m src.export   --run-name b0_mnv3small_frozen
```



## Technology Stack

| Area              | Choice                                                              |
| ----------------- | ------------------------------------------------------------------- |
| ML                | TensorFlow/Keras, tf.data, NumPy, pandas, scikit-learn, matplotlib  |
| Backbone (start)  | MobileNetV3-Small or -Large (Keras Applications)                    |
| Later comparisons | EfficientNetV2-B0, MobileNetV2                                      |
| Experiments       | `results.csv` (add W&B only if you want)                            |
| Export            | `tf.lite.TFLiteConverter` (float16, then int8)                      |
| Mobile            | Flutter + `tflite_flutter` package, `image_picker` (camera/gallery) |



## Repository Structure

```
bird-classifier/
├── configs/            # small YAML/JSON per experiment
├── data/
│   ├── raw/            # git-ignored
│   └── splits/         # train.csv, val.csv, test.csv (path,label)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_baseline.ipynb
├── src/
│   ├── data.py         # load CSVs, build tf.data pipelines, augmentation
│   ├── model.py        # build_model(backbone, num_classes)
│   ├── train.py
│   ├── evaluate.py
│   └── export.py       # Keras -> TFLite + parity check
├── experiments/results.csv
├── models/             # bird_v1.tflite, labels.txt, model_config.json
├── mobile/             # Flutter app
├── requirements.txt
├── .gitignore
└── README.md
```

