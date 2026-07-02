This folder contains two implementations of the collision checking and scoring pipeline.

"streaming_collision_check.cu" uses multiple streams and events to allow for many different operations to occur in parallel.
The collision check, trajectory scoring, and various memory operations can happen alongside one another. Events are used to express dependancies.

"serial_collision_check.cu" runs everything using basic, syncronous APIs. Everything happens in series.

When warmed up and averaged over 10 runs, the streaming approach takes 125ms to run while the serial approach takes 215ms.

There are two .png files which show the overlap/non overlap inside nsys-ui