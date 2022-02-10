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


## Schedule

Feb 1
* Fix preprocess
* Add back double network implementation
* Figure out why crash
* Update Q-value differently according to `done` flag
  * Passing the `done` flag to the replay memory when a life is lost (without resetting the game)

Feb 8
* Fix memory issue
* Add back A3C implementation
* Continue implementing PPO
* Skeleton of thesis, subsections with bullet points
* Model for Wildfire domain with openness

Feb 15
* Start writing sections of thesis
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