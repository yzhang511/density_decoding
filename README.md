<a name="readme-top" id="readme-top"></a>

<!-- PROJECT LOGO -->

<div width="150" align="right">
<a href="https://github.com/yzhang511/density_decoding">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/yzhang511/density_decoding/blob/main/images/icon.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/yzhang511/density_decoding/blob/main/images/icon.png">
      <img alt="Logo toggles light and dark mode" src="https://github.com/yzhang511/density_decoding/blob/main/images/icon.png"  width="150" align="right">
    </picture>
</a>
</div>
## Density-Based Spike Sorting-Free Decoding


We implemented a spike sorting-free decoding method that directly models the distribution of extracted spike features using a mixture of Gaussians (MoG) encoding the uncertainty of spike assignments, without aiming to solve the spike clustering problem explicitly. We allow the mixing proportion of the MoG to change over time in response to the behavior and develop variational inference methods to fit the resulting model and to perform decoding. 

We benchmark our method with an extensive suite of recordings from different animals and probe geometries, demonstrating that our proposed decoder can consistently outperform current methods based on thresholding (i.e. multi-unit activity) and spike sorting.

**[NeurIPS 2023 (spotlight)]** [Bypassing spike sorting: Density-based decoding using spike localization from dense multielectrode probes](https://www.biorxiv.org/content/10.1101/2023.09.21.558869v1)


## Installation
Create an environment and install this package and any other dependencies:
```
conda create --name density_decoding python=3.8
conda activate density_decoding
git clone https://github.com/yzhang511/density_decoding.git
cd density_decoding
pip install -e .
```

## Usage
Example usage can be found with `tutorials.ipynb` and `in_depth_tutorials.ipynb`. In `tutorials.ipynb`, we demonstrate decoding using custom datasets or IBL public dataset. For a custom dataset, user should provide spike features obtained from [this pipeline](https://github.com/int-brain-lab/spikes_localization_registration) and corresponding behavior over time to this decoder. For the public IBL dataset, please download it directly through the ONE API and provide it to this model. 

In `in_depth_tutorials.ipynb`, we mainly focus on IBL public dataset and performed standard analysis presented in the paper. Additionally, we include clusterless unsupervised analysis and provide visualizations of the firing rates $\lambda$, weight matrix $W$ as well as $\beta, U, V$ matrices which are learned through variational inference.
