# fastMRI

At the time of research, the fastMRI code repo/library is licensed under an MIT 
license which has been copied into this directory under LICENSE.md.

The original code is available at https://github.com/facebookresearch/fastMRI

We make a number of necessary modifications that extend and modify some of the 
original behavior.

- new non-rectangular mask functions and changes to mask data types / schemas
    * Poisson mask
    * Radial mask
    * Gaussian mask
- updated versions of certain packages (wandb, etc.)
- working image logging
- documentation
- type annotations

## Cite

Cite the original fastMRI arXiv paper:

```BibTeX
@misc{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Tullie Murrell and Zhengnan Huang and Matthew J. Muckley and Aaron Defazio and Ruben Stern and Patricia Johnson and Mary Bruno and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and Nafissa Yakubova and James Pinkerton and Duo Wang and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}
```
