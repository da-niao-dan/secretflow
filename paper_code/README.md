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

benchmark multiparty run

```bash
python  multiparty_run.py --n 2 --m 2000 > log.txt 2>&1
python benchmark_parser.py --log_file log.txt --n_clients 2 --n_inputs 2000
```

Example output:

```bash
N_Clients=2 N_Inputs=2000 Time_Preprocessing=0.684328 Sent_Preprocessing=104 Time_Local_Commitment=0.000994 Sent_Local_Commitment=104 Time_Cosine_Similarity=0.180873 Sent_Cosine_Similarity=331 Time_Euclidean_Norm=0.7512 Sent_Euclidean_Norm=220 Time_Meta_Clipping=0.246202 Sent_Meta_Clipping=104 Time_Aggregation=0.372031 Sent_Aggregation=104 Time_Total=2.235628 Sent_Total=967
```

Note that in order to correctly log the bytes sent, edit the .env file in this folder to be

```bash
ENABLE_COMMUNICATION_LOG=True
```

In order to gain best run time performance, we need to disable the log function, edit the .env file in this folder to be

```bash
ENABLE_COMMUNICATION_LOG=False
```
