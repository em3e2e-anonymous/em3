# EM3 (Based On MMRec)


Thanks to the open-source framework [MMRec](https://github.com/enoche/MMRec) , we can implement our EM3 at a very low cost.

## Project
We mainly add 3 files:
* `src/models/em3.py`: adds our Fusion and CIC modules into FREEDOM model.
* `src/configs/model/EM3.yaml`: contains many hyper-parameters.
* `src/common/transformer.py`: contains the implementation of FQ-Former.

## Datasets
Download from Google Drive: [Baby/Sports](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing).

The data already contains text and image features extracted from Sentence-Transformers and CNN.

* Please move your downloaded data into this dir for model training.

## Method
run following commands and the results will be logged in `src/log`
```
cd src
python3 main.py -m EM3 -d baby
```
