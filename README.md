# Paper Results

This branch contains the results for our publication. The following experiments can be found in the `results` directory:

- The results from Table 1 can be obtained by running the `PneumoniaClassification.py` script
- The results from Table 2 can be obtained by running the `LiverSegmentation.py` script
- The results from Table 3 can be obtained by running the scripts contained in the following directories:
    - The timing benchmarks can be found in `benchmarks.py`
    - The memory benchmarks (and additional timing benchmarks) can be found in the `*_benchmarks_memory` directories. The `unsupported_layer_benchmarks` directory contains an additional module `unet.py` which is required as an import but doesn't contain any experimental code.
    

The `dataloader` script contains utilities and is not required.

Running these scripts requires the following additional dependencies: `scikit-learn`, `albumentations`, `opacus`, `pyvacy`, `tqdm` and `pytorch-lightning`. Optionally, `tensorboard` can be used to monitor trainings.
