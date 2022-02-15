# Log

`$dmeseg`
* `[599691.745183] python3 invoked oom-killer: gfp_mask=0x100cca(GFP_HIGHUSER_MOVABLE), order=0, oom_score_adj=0`
* `[599691.745469] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,global_oom,task_memcg=/user.slice/user-10016.slice/user@10016.service,task=python3,pid=1478466,uid=10016`
* `[599691.745499] Out of memory: Killed process 1478466 (python3) total-vm:76128320kB, anon-rss:62511932kB, file-rss:72584kB, shmem-rss:72kB, UID:10016 pgtables:134324kB oom_score_adj:0`

`$htop`
* MEM% full
* Around 90%, OOM kill gnome display manager, python3
* How to locate memory consuming objects
* `torch.cuda.empty_cache()`

`tail -f /var/log/syslog` - permission denied

## A3C
* Asynchronous advantage actor-critic for each actor learner thread
  * Parameters
    * global shared parameter theta, theta_v
    * global shared counter T = 0
    * thread parameter theta', theta_v'
  * Initialize thread counter t = 1
  * Repeat:
    * Reset gradients: for theta, theta_v
    * Synchronize: theta' = theta, theta_v' = theta_v
    * t_start = t
    * Get state s_t
    * Repeat:
      * Take action a_t according to policy pi(a_t|s_t; theta')
      * Get reward r_t and next state s_t+1
      * t = t + 1
      * T = T + 1
    * Until: s_t is terminal or t - t_start == t_max
    * R = 0, for terminal s_t
    * R = V(s_t, theta_v') for non-terminal s_t // bootstrap from last state
    * for i from (t - 1) to t_start do
      * R = r_i + gamma * R
      * Accumulate gradients wrt theta'
      * Accumulate gradients wrt theta_v'
    * end for
    * Perform asynchronous update of theta and theta_v using gradients
  * Until: T > T_max
  
* Observations (CNN)
  * Shared non-output layers
  * Policy net (actor) -> pi(a|s), with one softmax output
  * Value net (critic) -> V(s), with one linear output

* Adding the entropy of the policy π to the objective function improved exploration by discouraging
premature convergence to suboptimal deterministic policies. 


## Questions
* LSTM?


## Schedule

Feb 1
* Fix preprocess
* Add back double network implementation
* Figure out why crash
* Update Q-value differently according to `done` flag
  * Passing the `done` flag to the replay memory when a life is lost (without resetting the game)

Feb 8
* Fix memory issue
* Add A3C implementation
* Continue implementing PPO
* Model for Wildfire domain with openness

- RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
- deadlock on res_queue.get()

Feb 15
* n-step loss
* Skeleton of thesis, subsections with bullet points
* Implement DQN on small Wildfire domain

Feb 22
* Writing sections of thesis

Feb 29
* writing sections of thesis
* Implement on Wildfire domain

Mar 8
* writing sections of thesis

Mar 15
* writing sections of thesis

Mar 22 
* Run experiments and analyze
  * With or without openness

Mar 29

Apr 5 (Spring break)
* Finish thesis

Apr 12