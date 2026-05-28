b = 1024
j = 8
e = 54000

num_compares = b * j * e

robot_sphere_idx = []
obs_idx = []
robot_idx = []

for i in range(num_compares - 64, num_compares):
    robot = i // (j * e)
    sphere = (i % (j * e)) // e
    obstacle = i % e

    sphere = robot * j + sphere
    
    robot_sphere_idx.append(sphere)
    obs_idx.append(obstacle)
    robot_idx.append(robot)

print(robot_sphere_idx)
print(obs_idx)
print(robot_idx)