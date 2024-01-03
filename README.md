# A Learning-Based Account of Phonological Tiers

**In Progress.** This will be the repository for the code for *A Learning-Based Account of Phonological Tiers* 

The code is already here, and I will be rolling out documentation on setting it up, running it, and extending it to your own data. Stay tuned!  

```bibtex
@article{belth2024tiers,
  title={A Learning-Based Account of Phonological Tiers},
  author={Belth, Caleb},
  journal={Linguistic Inquiry},
  note={In Press},
  year={2024},
  publisher={MIT Press}
}
```

## Usage Examples

All examples assume that you are running from the ```src/``` directory. If this is not the case, you can add the ```src/``` directory to the current path, as shown below, before importing the code.

```python
>> import sys
>> sys.path.append(path_to_src)
```

### Loading Data

Below is an example of how to load the Turkish CHILDES corpus.

```python
>> from utils import load
>> pairs, freqs = load('../data/turkish/childes.txt', skip_header=True)
```

### Running D2L

Here is an example of running D2L on the first 1K words from the Turkish CHILDES corpus. See above for loading (UR, SR) pairs.

```python
>> from plp import PLP
>> plp = PLP(ipa_file='../data/german/ipa.txt')
>> plp.train(pairs[:1000])
>> print(plp)
1: {+voi,-son} --> [-voi] /  __ .
```

### Using Generalizations

Once PLP is trained, as exemplified above, it can be used to map a UR to an SR. To apply all the rules in the grammar at once, you can simply call the model object on an input:

```python
>> plp('hʊnd.')
'hʊnt.'
```

If you want to apply rules individually, you can access them as follows:

```python
>> plp.grammar.rules
[{+voi,-son} --> [-voi] /  __ .]
>> r = plp.grammar.rules[0]
>> r('hʊnd.')
'hʊnt.'
```