## DPMMSubClustersStreaming

This package is a Python wrapper for the [DPMMSubClustersStreaming.jl](https://github.com/BGU-CS-VIL/DPMMSubClustersStreaming.jl) Julia package.<br>

<br>
<p align="center">
<img src="appended.gif" alt="Streaming DPGMM">
</p>


### Installation

1. Install Julia from: https://julialang.org/downloads/platform
2. Add our DPMMSubClusterStreaming package from within a Julia terminal via Julia package manager:
```
] add DPMMSubClustersStreaming
```
3. Add our dpmmpythonStreaming package in python: pip install dpmmpythonStreaming
4. Add Environment Variables:
	#### On Linux:
	1. Add to the "PATH" environment variable the path to the Julia executable (e.g., in .bashrc add: export PATH =$PATH:$HOME/julia/julia-1.6.0/bin).
	#### On Windows:
	1. Add to the "PATH" environment variable the path to the Julia executable (e.g., C:\Users\<USER>\AppData\Local\Programs\Julia\Julia-1.6.0\bin).
5. Install PyJulia from within a Python terminal:
```
	import julia;julia.install();
```

### Usage Example:

```
from dpmmpythonStreaming.dpmmwrapper import DPMMPython
from dpmmpythonStreaming.priors import niw
import numpy as np
from julia.api import Julia
jl = Julia(compiled_modules=False)
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
