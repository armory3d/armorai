
**Setup**

```bash
conda env create -f environment_train.yml
conda activate inpaint_train
```

```bash
python train.py
```

```bash
conda env create -f environment_test.yml
conda activate inpaint_test
```

```bash
python test.py
```
