# BKA-python

To run one case:
```shell
# for example:
python main.py --dataset CEC2005 --fuction F1 --iteration 2000 --repeat_times 1 --search_agents 30
```

To run whole CEC2005 dataset:

```shell
./RUN_CEC2005.sh
```

The result will be stored in 'output/{args.dataset}/{args.function}'
