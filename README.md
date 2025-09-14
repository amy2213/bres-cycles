# BRES Cycle Detection Framework

A Bayesian symbolic cycle detector with built-in datasets for **Venus** (Dresden) and **Coligny**.

## Quickstart
- Venus daily:
```bash
bres cycles detect --dataset venus --venus-mode daily --venus-cycles 65   --candidates 584,2920,37960 --report-supercycle
```
- Venus phase:
```bash
bres cycles detect --dataset venus --venus-mode phase --venus-cycles 100   --candidates 4,8,12,36
```
- Coligny 5-year plaque:
```bash
bres cycles detect --dataset coligny --coligny-mode month --coligny-years 5   --candidates 12,13,31,62,124 --report-supercycle
```


## Testing

```bash
pip install -e .
pip install pytest
pytest
```
