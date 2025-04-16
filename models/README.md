## Neural Operator Models
udno.py : U-shaped DISCO Neural Operator
    * in-place resolution invariant replacement for U-Net
    * building block for no_varnet
    * uses EquidistantDiscreteContinuousConv2d from torch_harmonics

no_varnet.py : Neural Operator model introduced in https://arxiv.org/abs/2410.16290


If you use either the no_varnet or UDNO, please cite the following.

```bibtex
@article{jatyani2024unified,
  title   = {A Unified Model for Compressed Sensing MRI Across Undersampling Patterns},
  author  = {Jatyani, Armeet Singh and Wang, Jiayun and Wu, Zihui and Liu-Schiaffini, Miguel and Tolooshams, Bahareh and Anandkumar, Anima},
  journal = {arXiv preprint arXiv:2410.16290},
  year    = {2024}
}
```

```bibtex
@article{liu2024neural,
  title={Neural operators with localized integral and differential kernels},
  author={Liu-Schiaffini, Miguel and Berner, Julius and Bonev, Boris and Kurth, Thorsten and Azizzadenesheli, Kamyar and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2402.16845},
  year={2024}
}
```

## E2E-Varnet
varnet.py : Original E2E-Varnet model

Please cite the original E2E-Varnet paper.

```bibtex
@inproceedings{sriram2020end,
  title={End-to-end variational networks for accelerated MRI reconstruction},
  author={Sriram, Anuroop and Zbontar, Jure and Murrell, Tullie and Defazio, Aaron and Zitnick, C Lawrence and Yakubova, Nafissa and Knoll, Florian and Johnson, Patricia},
  booktitle={Medical image computing and computer assisted intervention--MICCAI 2020: 23rd international conference, Lima, Peru, October 4--8, 2020, proceedings, part II 23},
  pages={64--73},
  year={2020},
  organization={Springer}
}
```
