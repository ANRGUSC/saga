# General Scheduler

## How this works:

The General Scheduler takes in three parameters:
<ol>
<li>A ranking heuristic like Upward Rank</li>
<li>A Schedule Type like append only or insertion. Given a task and a node, it decides how to insert the task into the node's schedule. This is a class, whose object is initialized everytime the function "schedule" on the General Scheduler is called. <br>
It has 3 methods-
<ul>
<li> insert: this function directly inserts the task into the schedule of the node.
<li> get_earliest_finish: this function returns the earliest time a task will finish if scheduled on the node
<li> get_earliest_start: this function returns the earliest time a task will start if scheduled on the node
</ul>
The last 2 funtions were added as some selection metrics decide how to schedule based on earliest finish-time/start-time
</li>
<li>A selection metric that decides how to scheduled ranked tasks using the Schedule Type. This function takes in the ranked tasks and the schedule type object as input</li>
</ol>