## DPMMSubClustersStreaming

This package is a Python wrapper for the [DPMMSubClustersStreaming.jl](https://github.com/BGU-CS-VIL/DPMMSubClustersStreaming.jl) Julia package.<br>

<br>
<p align="center">
<img src="appended.gif" alt="Streaming DPGMM">
</p>


### Installation

```
pip install dpmmpythonStreaming
```

If you already have Julia installed, install [PyJulia](https://github.com/JuliaPy/pyjulia) and add the package `DPMMSubClustersStreaming` to your julia installation. <p>
<p>
Make sure Julia path is configured correctly, e.g. you should be able to run julia by typing `julia` from the terminal, unless configured properly, PyJulia wont work.


**Installation Shortcut for Ubuntu distributions** <br>
If you do not have Julia installed, or wish to create a clean installation for the purpose of using this package. after installing (with pip), do the following:

```
import dpmmpythonStreaming
dpmmpythonStreaming.install()
```
Optional arguments are `install(julia_download_path = 'https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.0-linux-x86_64.tar.gz', julia_target_path = None)`, where the former specify the julia download file, and the latter the installation path, if the installation path is not specified, `$HOME$/julia` will be used.<br>
As the `install()` command edit your `.bashrc` path, before using the pacakge, the terminal should either be reset, or modify the current environment according to the julia path you specified (`$HOME$/julia/julia-1.4.0/bin` by default).

### Usage Example:

```
from dpmmpythonStreaming.dpmmwrapper import DPMMPython
from dpmmpythonStreaming.priors import niw
import numpy as np

j = julia.Julia(compiled_modules=False)
data,gt = DPMMPython.generate_gaussian_data(10000, 2, 10, 100.0)
batch1 = data[0:5000]
batch2 = data[5000:]
prior = DPMMPython.create_prior(2, 0, 1, 1, 1)
model= DPMMPython.fit_init(batch1,100.0,prior = prior,verbose = True, burnout = 5, gt = None, epsilon = 1.0)
labels = DPMMPython.get_labels(model)
model =fit_partial(model,1, 2, batch2)
labels = DPMMPython.get_labels(model)
print(labels)
```
### Misc

For any questions: dinari@post.bgu.ac.il

Contributions, feature requests, suggestion etc.. are welcomed.

If you use this code for your work, please cite the following:

```
@inproceedings{dinari2022streaming,
  title={Sampling in Dirichlet Process Mixture Models for Clustering Streaming Data},
  author={Dinari, Or and  Freifeld, Oren},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2022}
}
```
