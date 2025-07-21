# Pipeline for calcium imaging of organoids

## Setting up the environment 

### Option 1: Conda environment

### Option 2: Docker container

## Running the pipeline

### Extracting tiffs from the .lif file

Optional step if your data is in a single .lif file. 

```
python extract_tiffs_from_lif.py /path/to/lif/file --output /path/to/output/folder
```

### Preprocessing tiffs

Applies a gaussian blur denoising and chops up video if there are dark frames that mark the addition of a chemical stimulus