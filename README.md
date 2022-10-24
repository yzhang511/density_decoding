# clusterless decoding
clusterless decoding of neural data using a mixture of Gaussians (MoG)

Create conda environment and install packages:
```
conda create -n clusterless python=3.8
conda install --file requirements.txt
```

For jupyter lab (need to install `nodejs`):
```
pip install jupyterlab jupytext
conda install -c "conda-forge/label/cf202003" nodejs
conda update nodejs
jupyter lab build
```

Then serve:
```
mkdir -p .jupter/lab/workspaces
JUPYTERLAB_WORKSPACES_DIR=.jupyter/lab/workspaces jupyter lab --no-browser --ip=0.0.0.0
```

