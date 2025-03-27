# Running method

Assume we are at secretflow repo

install modified sf

```bash
pip install -e .
```

run the paper code:

```bash

cd paper_code
python FL_program.py > log.txt 2>&1

export ENABLE_COMMUNICATION_LOG=True
python FL_program.py > log.txt 2>&1
```

look at the log file at `paper_code/log.txt`.
