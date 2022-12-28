```
 __                __    __           _     _ 
/ _\_   _ _ __ ___/ / /\ \ \___  _ __| | __| |
\ \| | | | '_ ` _ \ \/  \/ / _ \| '__| |/ _` |
_\ \ |_| | | | | | \  /\  / (_) | |  | | (_| |
\__/\__, |_| |_| |_|\/  \/ \___/|_|  |_|\__,_|
    |___/              by Machine Teaching @ MPI-SWS 
                                                                 
```


## The Decision Process

### Code
A Code object executed on a SymWorld object using the FastEmulator is considered 
correct. The SymWorld will try to take the decisions accordingly, so that the 
emulator does not crash.

However, there are cases when the Code is inherently wrong (e.g., pick a marker up 
after the marker sensing returned a negative result). In this case, the SymWorld 
will make the emulator crash.

### Unknowns and Original_Markers
The SymWorld has cells marked as **unknown**, part of which will be populated 
through the decision process resulted from the execution of the given code. If the 
concerning cell was never an **unknown** cell, the SymWorld will behave as a normal 
World.

If the agent moves to an **unknown** cell, the SymWorld, it will not be unknown 
anymore, but the marker information is still missing. The marker decisions are 
reflected in the help of **original_markers**. A negative value means that it will 
be locked (i.e., it should no longer be changed via decisions). -0.25 will be 
replaced by a 0 and 0.5 by a 1. They cover corner cases related to ```markersPresent```.

### Pre- and Post-Grid
Note that the SymWorld handles the creation of both the pregrid World and the 
postgrid World at the same time (i.e., the differences are related to the markers and 
the hero).

### The Decision Process
There are various types of actions, based on the decisions the SymWorld has to make:
1. **No Decision**:
   - ```turnLeft```
   - ```turnRight```
2. **Straightforward Decision**: 
   - ```move``` - will make the **unknown** cell in front of the hero a free space
   - ```pickMarker``` - if the **original_markers** is not fixed, add 1 to it and 
     make markers 0, otherwise behave as in a normal World
   - ```putMarker``` - fix the **original_markers** to 0 (i.e., using -0.25) and 
     increase the markers
3. **Complex Decision** - employs a DecisionMaker:
   - ```frontIsClear```, ```leftIsClear```, ```rightIsClear``` - if the sensed 
     marker is within the grid, it will use the DecisionMaker to get a binary 
     decision and set the cell as blocked or not, accordingly, also clearing the 
     **unknown**
   - ```markersPresent``` - if no markers are already there and the **original_markers** 
     is not fixed, it will use the DecisionMaker to get a binary decision and set 
     the markers accordingly:
     - if the decision is positive, in case there are no **original_markers** (i.e., 0)
       it will fix the **original_markers** to 1 (i.e., using 0.5) and set the 
       markers to 1
     - otherwise, it will set the markers to 0 and fix the **original_markers** to 0 
       (i.e., using -0.25), if there were no original markers before, or it will fix 
       the **original_markers** to the existing value, by negation
   
### Initialization
The DecisionMaker is also capable to set the hero initial position and direction. If 
the position is not initialized, _any call to an action_ will initialize it by 
calling the DecisionMaker. If the direction is not initialized, only calls to 
```frontIsClear```, ```leftIsClear```, ```rightIsClear```, ```move```, 
```turnLeft```, and ```turnRight``` (i.e., no ```markersPresent```, ```pickMarker```,
and ```putMarker```) will initialize it by calling the DecisionMaker.