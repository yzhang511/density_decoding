<a name="readme-top" id="readme-top"></a>

<!-- PROJECT LOGO -->

<div width="100" align="right">
<a href="https://github.com/yzhang511/density_decoding">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/yzhang511/density_decoding/blob/main/assets/icon.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/yzhang511/density_decoding/blob/main/assets/icon.png">
      <img alt="Logo toggles light and dark mode" src="https://github.com/yzhang511/density_decoding/blob/main/assets/icon.png"  width="100" align="right">
    </picture>
</a>
</div>

## Density-Based Spike Sorting-Free Decoding
**[Motivation]** Neural decoding and its applications to brain computer interfaces (BCI) are essential for understanding the association between neural activity and behavior. A prerequisite for many decoding approaches is spike sorting, the assignment of action potentials (spikes) to individual neurons. Current spike sorting algorithms, however, can be inaccurate and do not properly model uncertainty of spike assignments, therefore discarding information that could potentially improve decoding performance.

**[Spike-feature based decoding]** Recent advances in high-density probes (e.g., Neuropixels) and computational methods now allow for extracting a rich set of spike features from unsorted data; these features can in turn be used to directly decode behavioral correlates:

<div align="right">
<a href="https://github.com/yzhang511/density_decoding">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/yzhang511/density_decoding/blob/main/assets/spike_localization_features.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/yzhang511/density_decoding/blob/main/assets/spike_localization_features.png">
      <img alt="Logo toggles light and dark mode" src="https://github.com/yzhang511/density_decoding/blob/main/assets/spike_localization_features.png"  align="right">
    </picture>
</a>
</div>

**[Method]** We propose a spike sorting-free decoding method that directly models the distribution of extracted spike features using a mixture of Gaussians (MoG) encoding the uncertainty of spike assignments, without aiming to solve the spike clustering problem explicitly. We allow the mixing proportion of the MoG to change over time in response to the behavior and develop variational inference methods to fit the resulting model and to perform decoding:

<div align="right">
<a href="https://github.com/yzhang511/density_decoding">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/yzhang511/density_decoding/blob/main/assets/model_diagram.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/yzhang511/density_decoding/blob/main/assets/model_diagram.png">
      <img alt="Logo toggles light and dark mode" src="https://github.com/yzhang511/density_decoding/blob/main/assets/model_diagram.png"  align="right">
    </picture>
</a>
</div>

**[Full paper]** [Bypassing spike sorting: Density-based decoding using spike localization from dense multielectrode probes](https://www.biorxiv.org/content/10.1101/2023.09.21.558869v1)

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## ⏳ Installation
Create an environment and install this package and any other dependencies:
```
conda create --name density_decoding python=3.8
conda activate density_decoding
git clone https://github.com/yzhang511/density_decoding.git
cd density_decoding
pip install -e .
```
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## ⚡️ Quick Start
Example usage can be found in [tutorials](https://github.com/yzhang511/density_decoding/tree/main/tutorials): 

0. `data_preprocessing`: Users can provide their own preprocessed spike features as decoder input. We also provide a tutorial for running the spike localization pipeline to extract high-resolution spatial features from Neuropixel probes. This notebook is under development ⚠️. For now, please refer to the legacy [IBL pipeline](https://github.com/int-brain-lab/spikes_localization_registration) or the current [pipeline](https://github.com/cwindolf/dartsort/blob/main/scripts/localize.py) under development for documentation on spike feature extraction.

1. `decoding_pipeline`: We demonstrate our decoding pipeline using custom datasets or IBL public datasets. For a custom dataset, user should provide their own spike features and behaviors as decoder inputs. For public IBL datasets, please download recordings directly through [ONE API](https://int-brain-lab.github.io/iblenv/notebooks_external/one_quickstart.html) and follow the notebook `data_preprocessing` to obtain spike features. 

2. `downstream_inference`: We use IBL public datasets as demo to show how to perform standard analysis in the paper. This notebook is a break-down of our decoding pipeline.

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## :computer: CLI

`density_decoding` comes with a quick [CLI](https://github.com/yzhang511/density_decoding/tree/main/CLI) decoding pipeline tool. 

If you have preprocessed IBL spike features, here is an example usage:
```
python decode_ibl.py --pid dab512bd-a02d-4c1f-8dbc-9155a163efc0 
--ephys_path PATH_TO_SPIKE_FEATURES --out_path OUTPUT_PATH
--brain_region ca1 --behavior choice --align_time_type stimOn_times 
--t_before 0.5 --t_after 1.0 --n_t_bins 30
```

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<!-- LICENSE -->
## :classical_building: License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## ✏️ Cite Us

If you found the paper useful, please cite us:
```
@article{
  zhang2023bypassing,
  title={Bypassing spike sorting: Density-based decoding using spike localization from dense multielectrode probes},
  author={Zhang, Yizi and He, Tianxiao and Boussard, Julien and Windolf, Charlie and Winter, Olivier and Trautmann, Eric and Roth, Noam and Barrell, Hailey and Churchland, Mark M and Steinmetz, Nicholas A and others},
  journal={bioRxiv},
  year={2023},
}
```
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

