# demo-random-search-RL

A toy demo based on the following paper: 

Mania, H., Guy, A., & Recht, B. (2018). Simple random search provides a competitive approach to reinforcement learning. Retrieved from http://arxiv.org/abs/1803.07055


### Method: 

The basic random search in Mania et al. 2018 (see alg 1), 
which is basically the <a href="https://en.wikipedia.org/wiki/Finite_difference_method">finite difference method</a>. 

### Results: 

Here's the learning curve on a 5x5 grid world, where the agent is trained on find the goal while avoiding the punishment. 

<img src="https://github.com/qihongl/demo-random-search-RL/blob/master/figs/lc.png" width='350'>


Here's a sample path on the grid world. 
- red dot: reward 
- black dot: punishment

<img src="https://github.com/qihongl/demo-random-search-RL/blob/master/figs/pg-path.png" width='300'>
