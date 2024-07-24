# BNAIC 2023 best paper award: "Laser Learning Environment: A new A new environment for coordination-critical multi-agent tasks"

This repo contains the code to reproduce the experiments of the paper ["Laser Learning Environment: A new A new environment for coordination-critical multi-agent tasks"](https://arxiv.org/abs/2404.03596) presented at the 2023 edition of the Benelux AI conference, in TU Delft. This paper has received the Best Paper Award.

The code of the environment itself can be found on [https://github.com/yamoling/lle](https://github.com/yamoling/lle).

## Running the experiments
To start an experiment, run
```bash
python src/main.py
```

To play with the parameters, you can overwrite the default values of the `Parameter`s in the `main.py` file. For instance, to use QMix with Prioritized Experience Replay and Random Network Distillation, you can write:
```python
params = Parameters(
    mixer="qmix",
    per=True,
    rnd=True
)
```

## Citing our work
The environment has been presented at [EWRL 2023](https://openreview.net/pdf?id=IPfdjr4rIs) and at [BNAIC 2023](https://bnaic2023.tudelft.nl/static/media/BNAICBENELEARN_2023_paper_124.c9f5d29e757e5ee27c44.pdf) where it received the best paper award.

```
@inproceedings{molinghen2023lle,
  title={Laser Learning Environment: A new environment for coordination-critical multi-agent tasks},
  author={Molinghen, Yannick and Avalos, Raphaël and Van Achter, Mark and Nowé, Ann and Lenaerts, Tom},
  year={2023},
  series={BeNeLux Artificial Intelligence Conference},
  booktitle={BNAIC 2023}
}
```