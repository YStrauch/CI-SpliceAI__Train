# Project Links
This repository is part of the [CI-SpliceAI](https://ci-spliceai.com) software package published in [PLOS One](https://doi.org/10.1371/journal.pone.0269159).

This is the project to train the models. You may also be interested in the [code to use trained models to annotate variants offline](https://github.com/YStrauch/CI-SpliceAI__Annotation), [code comparing different tools on variant data](https://github.com/YStrauch/CI-SpliceAI__Comparison), and the [website providing online annotation of variants](https://ci-spliceai.com).

# Setup

We strongly advise you to install [conda](https://docs.conda.io/en/latest/) and create a new conda environment. This keeps your projects and their dependencies separate. So go ahead and install conda first.

## Create environment with GPU support
Our setup was trained on two GTX 1060 TIs with CUDA 11.0 - the versions pinned here work for our architecture; you might need to change cudnn and tensorflow versions (check out https://www.tensorflow.org/install/source_windows#gpu).

```bash
conda create --yes -n cis python=3 keras=2.0.5 tensorflow-gpu=1.4.1 cudnn=7.0.5 matplotlib "h5py=<3=mpi*" numpy requests pandas pyfaidx mpi4py tensorboard scikit-learn -c bioconda -c conda-forge
```

**Notes**:
- You can substitute tensorflow-gpu with tensorflow for local development without a GPU.
- Using newer versions of keras or tensorflow worsened convergence, so careful when upgrading!
- Do not use h5py version 3 or you won't be able to load h5 models
- You can remove the "=mpi*" suffix from h5py, however then you can't use the multi-CPU "mpiexec" command later; instead run python directly
- There are some runtime warnings from mpi4py, which can be ignored.
- Do not update/unpin keras/tensorflow/cuda, newer versions will either make the model not converge during training or produce bad scores (I don't know why!)

# Project Structure

All scripts are to be run in the `cis` environment (run ``conda activate cis`` before running the script).
## splice_table.py
This script re-creates the splicing tables checked in the [data/](data/) folder. Since all splicing tables are already checked in, you don't need to run it for the default behaviour of collapsed isoforms.

## data/splicing_*
There is one train table and two annotation tables. The annotation tables are used in the CI-SpliceAI python module in order to annotate the most significantly affected gene. The train file combines genes in proximity to another to not create disagreeing ground truth files.
## create_data.py
This script creates the machine learning data. You need to run this first before training the model.
It is recommended to start this with multiprocessing enabled like this (only if you installed h5py with mpi support as suggested above):
```mpiexec -n <num-processes> python create_data.py```

## train.py <index> <fold>
Trains the model. You need to run create_data.py first (see above)!
Index should be 1-5 to train the respective model.
Fold is either ALL or TEST; ALL trains on all chromosomes and TEST excludes the test data (see TEST_FOLD in env.py for a list of test chromosomes)

Depending on your distribution, you may need to specify to use tensorflow instead of theanos (run ``export KERAS_BACKEND=tensorflow; python train.py <index>``)

## aggregate.py
This aggregates and optimises the five models into one frozen graph that takes the majority vote of the five CNNs. Train your models first.
The resulting file, `models/ALL/CI-SpliceAI.pb`, is the one ready to be used with the CI-SpliceAI python inference module.

By default, the script assumes you want to build an ensemble trained on ALL chromosomes, and packages models with indices 1,2,3,4,5 together. You can change this; running with no parameters is equal to this command

```python aggregate.py --models=1,2,3,4,5 --folder models/ALL --output models/ALL/ensemble_frozen.pb```

## test.py
Tests models trained on TRAIN fold on the remainder of data.

# Paralog annotations
Paralog annotations are used only when using a train/test split, to exclude paralogs from the test split. They are not used when training the final models on all chromosomes.

The file [data/paralogs_GRCh38.csv](data/paralogs_GRCh38.csv) was created by Ensembl biomart on 30/03/22. Follow these steps to re-create the file with the newest Ensembl data:

1) Click [this link](https://www.ensembl.org/biomart/martview/6933d68650e7338265ebb6ed51e7cba2?VIRTUALSCHEMANAME=default&ATTRIBUTES=hsapiens_gene_ensembl.default.homologs.chromosome_name|hsapiens_gene_ensembl.default.homologs.ensembl_gene_id|hsapiens_gene_ensembl.default.homologs.hsapiens_paralog_orthology_type&FILTERS=hsapiens_gene_ensembl.default.filters.chromosome_name."1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y"&VISIBLEPANEL=resultspanel). If the page loooks wonky you might need to refresh.
0) Change the left dropwdown to _Compressed web file (notify by email)_
0) Ensure that the second dropdown is on _CSV_
0) Ensure that _Unique results only_ box is checked
0) Enter your email address
0) Click the _Go_ button
0) New text should have popped up at the bottom telling you that it is created in the background; wait for the email.

# License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.