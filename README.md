# About
Evolutionary Algorithms are used to bring the solutions closer to global minimum each steps. This repository implements two algorithms that are based on Differential Evolution (DE).
- Composite Differential Evolution (CoDE)
- Adaptive Differential Evolution (JADE)

## Installation
If `sourcedefender` is not installed,
```bash
pip install sourcedefender
```

## Usage
Choose between **Random Search**, **CoDE** or **JADE**.
```python
# self.RandomSearch(maxFES=FES)
self.CoDE(maxFES=FES)
# self.JADE(maxFES=FES)
```
Run with `python3`.
```bash
python3 globaloptimization.py
```

## Results
| Function | Random Search |   CoDe    |    JADE   | Function evaluation |
|----------|---------------|-----------|-----------|---------------------|
|     1    |      0.021    | 9.965e-09 | 8.052e-09 |        1000         | 
|     2    |      0.322    | 7.024e-30 | 6.546e-13 |        1500         | 
|     3    |     12.867    | 1.230e-06 | 3.752e-05 |        2000         | 
|     4    |     48.735    |    0.5344 |     0.187 |        2500         |

## References
- https://ieeexplore.ieee.org/document/5688232
- https://ieeexplore.ieee.org/document/5208221