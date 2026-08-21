# Performance Numbers
Base: 550.60us
WMMA: 755.40us

This problem is heavily memory-constrained and WMMA adds overhead to load 16x16 matrices so we can perform 6x7.7x8 multiplication. Using WMMA here does not fit well.

# Precision Requirements for Motion Control
I am going to discuss apply this to small to medium sized mobile robotic movement, like a roomba or a indoor delivery system. These are the systems that I am most familiar with.

Motion planning for this kind of system uses numbers in the 0-50, maybe 100 range. Robots do not typically travel at speeds higher than a few meters per second, planning horizons are a few seconds, accelerations are kept low when possible. This is important when discussing the precision requirements of floating point numbers since numbers with a higher magnitude have lower precision after the decimal.

Let's take a case where a robot is traveling at 2 m/s, and we want to plan movement out to 30 seconds. In these ranges we have about 2 points of precision after the decimal for FP16 and TF32. So if we are trying to represent a value near 2 m/s we might have plus or minus ~0.01 m/s of error.

## Robot Controllability
We do not have perfect control of the robot. In many systems we could easily have 5% error between the command velocity and real velocity. If we command 2 m/s we can get between 1.9-2.1 m/s. If the error comes from something like mis-calibration of wheel diameter or wheel slippage it will be a bias that does not average out over time.

## Ground Truth Planning
If we travel forward at 2 m/s for 30 seconds we will travel 60 meters.

## Planning with Precision Error
If our low precision representation instead computes 2.01 m/s * 30.01s we will predict travel of 60.32 m

The possible error induced by FP16 is 0.32 meters. 

## Planning with Imperfect Control
If our robot actually travels at 2.1 m/s: 2.1 m/s * 30s = 63 m

The possible error for our robot's actual trajectory is 3m.

We can see here that FP16 and TF32 provide sufficient precision for the kinds of planning that we might require for simple mobile robotics. That being said, we are only an order of magnitude away from the precision having an impact. A faster robot, a more complex path, fast accelerations, etc. could easily have an impact on planning. I would use FP32 as my default representation for robotics and reach for FP16 only if there is a clear need, and test extensively for my use case.  