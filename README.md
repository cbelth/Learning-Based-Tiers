# A Learning-Based Account of Phonological Tiers

Code for the 2024 *Linguistic Inquiry* paper *A Learning-Based Account of Phonological Tiers*.

```bibtex
@article{belth2024tiers,
  title={A Learning-Based Account of Phonological Tiers},
  author={Belth, Caleb},
  journal={Linguistic Inquiry},
  year={2024},
  publisher={MIT Press},
  url = {https://doi.org/10.1162/ling\_a\_00530},
}
```

## Reproducing Results

The results are in the `results/` directory. If you wish to reproduce them, please see the script `exp.py` in the `experiments` directory. The script takes two arguments:

```
--exp-name / -e, type=str (one of Turkish-CHILDES|Turkish-Morpho|Finnish|Latin)
--model / -m type=str, (one of D2L|GR|trigram|TSLIA|GG|LSTM)
```

`exp-name` specifies which dataset to produce the results for. `model` specifies which model to run. The results are saved in the directory `results/{exp_name}/{model}`.

## Running on Your Own Data

D2L will soon be implemented in the Python package [algophon](https://github.com/cbelth/algophon). This will make it easy to run the model on your own data. 

Until then, please see the `load` function in `utils.py` for how to load data, and `d2l.py` for the entry point to running D2L. For example data formats, see the files in `data/`.