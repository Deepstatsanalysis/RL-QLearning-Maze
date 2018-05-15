# Reinforce Learning Practice
Maze

## Prerequisite
- Python 3.6.4

## Install Dependency
```sh
$ pip install -r requirements.txt
```

## Usage
```sh
$ python usage: main.py [-h] [-l LENGTH] [-i ITERATION]
```

| optional Options           | Description                                    |
| ---                        | ---                                            |
| -h, --help                 | show this help message and exit                |
| -l LENGTH                  | input the length of the grid                   |
| -i ITERATION               | input the iteration of training                |


## Game Rules

\_ \_ \_ \* <- Destination, will get +1 reward <br>
\_ \_ \_ \_<br>
\_ \_ \_ \_<br>
o \_ \_ \_<br>
^<br>
Start Point<br>

We can choose 'up', 'down', 'left' and 'right' to approach destination

## Algorithm
- Q-Learning
  - Initialize Q-Table:

| state  | 'up'    | 'down'  | 'left'  | 'right' |
| ---    | ---:    | ---:    | ---:    | ---:    |
| (0, 0) |     0.0 |     0.0 |     0.0 |     0.0 |
| (0, 1) |     0.0 |     0.0 |     0.0 |     0.0 |
| (0, 2) |     0.0 |     0.0 |     0.0 |     0.0 |
| (0, 3) |     0.0 |     0.0 |     0.0 |     0.0 |
| (1, 0) |     0.0 |     0.0 |     0.0 |     0.0 |
| (1, 1) |     0.0 |     0.0 |     0.0 |     0.0 |
| (1, 2) |     0.0 |     0.0 |     0.0 |     0.0 |
| (1, 3) |     0.0 |     0.0 |     0.0 |     0.0 |
| (2, 0) |     0.0 |     0.0 |     0.0 |     0.0 |
| (2, 1) |     0.0 |     0.0 |     0.0 |     0.0 |
| (2, 2) |     0.0 |     0.0 |     0.0 |     0.0 |
| (2, 3) |     0.0 |     0.0 |     0.0 |     0.0 |
| (3, 0) |     0.0 |     0.0 |     0.0 |     0.0 |
| (3, 1) |     0.0 |     0.0 |     0.0 |     0.0 |
| (3, 2) |     0.0 |     0.0 |     0.0 |     0.0 |
| (3, 3) |     0.0 |     0.0 |     0.0 |     0.0 |

  - Initialize Enviroment and get current state, s
  - According to s, Actor will give an action: (ε-Greedy, e.g. ε = 0.9)
    - 10%: random choose one of 'up', 'down', 'left' or 'right'
    - 90%: choose the action with the highest Q-value
  - Take the action, and observe the reward, r, as well as the new state, s'.
  - Update the Q-table for the state using the observed reward and the maximum reward possible for the next state.
    - ![Q(s,a)=Q(s,a)+\alpha(r+\gamma\max_{a'}Q(s',a')-Q(s,a))](https://latex.codecogs.com/svg.latex?Q%28s,a%29=Q%28s,a%29+\alpha%28r+\gamma\max_{a%27}Q%28s%27,a%27%29-Q%28s,a%29%29)= Q(s, a) + \alpha (r + \gamma \max{_x}(Q(s', x)) - Q(s, a)))
  - Set the state to the new state, and repeat the process until a terminal state is reached.

## Performance
### 4 * 4 Grid

- Training

| Episode |   0 |  1  |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |  10 |  11 |  12 |  13 |  14 |  15 |  16 |  17 |  18 |  19 |
| ---     | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| Step    |  43 | 99  |  60 |  41 |  30 |  23 |  12 |  28 |   6 |   6 |  10 |   6 |   8 |   6 |   6 |   6 |   7 |  11 |  6 |   6 |

- Q-Table
     
| state  | 'up'     | 'down'       | 'left'       | 'right'      |
| ---    | ---:     | ---:         | ---:         | ---:         |
| (0, 0) | 0.000002 | 6.561000e-06 | 2.131669e-06 | 4.287023e-05 |
| (0, 1) | 0.000091 | 6.541390e-04 | 5.867699e-06 | 3.249000e-02 |
| (0, 2) | 0.039051 | 8.100000e-04 | 8.100000e-04 | 3.439000e-01 |
| (0, 3) | 0.000000 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| (1, 0) | 0.000002 | 5.904900e-07 | 1.246590e-05 | 3.055969e-03 |
| (1, 1) | 0.001654 | 3.490452e-05 | 3.018060e-05 | 1.744149e-01 |
| (1, 2) | 0.039780 | 7.900173e-04 | 5.147395e-03 | 4.045116e-01 |
| (1, 3) | 0.814698 | 2.195100e-03 | 8.580411e-03 | 3.249000e-02 |
| (2, 0) | 0.000012 | 0.000000e+00 | 5.904900e-07 | 2.681199e-04 |
| (2, 1) | 0.062038 | 1.254579e-05 | 5.904900e-07 | 1.385100e-04 |
| (2, 2) | 0.000810 | 3.431403e-05 | 2.203318e-04 | 1.389849e-02 |
| (2, 3) | 0.150792 | 1.975590e-04 | 7.290000e-05 | 2.195100e-03 |
| (3, 0) | 0.000001 | 1.816146e-06 | 1.396565e-06 | 4.158873e-03 |
| (3, 1) | 0.017853 | 6.348465e-04 | 9.121653e-07 | 2.827266e-07 |
| (3, 2) | 0.000251 | 0.000000e+00 | 3.141407e-06 | 0.000000e+00 |
| (3, 3) | 0.010342 | 0.000000e+00 | 0.000000e+00 | 1.975590e-04 |

## Related Link
- [nbviewer](https://nbviewer.jupyter.org/github/yutongshen/DSAI-HW3-Subtractor/blob/master/Subtractor.ipynb)

## Authors
[Yu-Tong Shen](https://github.com/yutongshen/)
