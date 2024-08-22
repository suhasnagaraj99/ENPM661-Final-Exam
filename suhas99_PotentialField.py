'''
This code file is part of submission for ENPM661 Final Exam.

Potential Field algorithm for path planning is implemented in the code.

The map ranges from -3000mm to 3000mm in x axis and -1500mm to 1500mm in the y axis with the origin at the centre.

The initial point is at (-3000,0) and the goal point is at (3000,0)
'''

import numpy as np
import matplotlib.pyplot as plt

# Defining grid dimensions
x_val = np.arange(-300, 301, 1)
y_val = np.arange(-150, 151, 1)
map_x, map_y = np.meshgrid(x_val, y_val)

## The starting / initial point (in cm)
initial = [-300, 0]

## The goal point (in cm)
goal = [300, 0] 

## range upto which the repulsive force is felt by the agent
force_range = 60

## Goal distance threshold
goal_thresh = 1

## Defining the obstacles in the map
## All the obstacles are circles and there are defined in the form (x_centre,y_centre,radius)
obstacles = [[-188, 92.5, 40], [-37, -60, 70], [145, 70, 37.5]]

## Attractive and repulsive coeffients that determine the attactive and repulsive forces felt by the agent
attraction_coefficient = 50
repulsion_coefficient = -200

## Creating arrays to hold force component values and initially filling it with zeros
x = np.zeros_like(map_x)
y = np.zeros_like(map_y)

## For each cell in the map (1cm x 1cm), the force is calculated based on the obstacle, goal and the coeffients
for i in range(map_x.shape[0]):
    for j in range(map_x.shape[1]):
        
        ## Distance to goal from the current point
        distance_goal = np.sqrt((goal[0] - map_x[i][j])**2 + (goal[1] - map_y[i][j])**2)
        
        ## Angle to goal from the current point
        angle_goal = np.arctan2(goal[1] - map_y[i][j], goal[0] - map_x[i][j])
        
        ## If the distance to goal is less than goal threshold (the points inside goal), the force is zero
        if goal_thresh > distance_goal:
            x[i][j] = 0
            y[i][j] = 0
        
        ## All points outside the goal_threshold feels an attractive force towards the goal    
        else:
            ## Magnitude of the attractive force. Farther the agent, more it is attracted to the goal.
            attractive_force_magnitude = attraction_coefficient * (distance_goal - goal_thresh)
            ## X and y components of the attractive force
            x[i][j] = attractive_force_magnitude * np.cos(angle_goal)
            y[i][j] = attractive_force_magnitude * np.sin(angle_goal)

        ## For each obstacle, the repulsive force (negative) is calculated and is added to the attractive force
        for obstacle in obstacles:
            
            ## Distance to the obstacle
            distance_obstacle = np.sqrt((obstacle[0] - map_x[i][j])**2 + (obstacle[1] - map_y[i][j])**2)
            
            ## Angle to the obstacle
            angle_obstacle = np.arctan2(obstacle[1] - map_y[i][j], obstacle[0] - map_x[i][j])
            
            ## If the distance is lesser than the radius + force range, the agent feels a repulsive force
            if distance_obstacle < obstacle[2] + force_range:
                
                ## Repulsive force magnitude
                repulsive_force_magnitude = repulsion_coefficient * (force_range + obstacle[2] - distance_obstacle)
                
                ## X and y components of the repulsive force force is added to the existing attractive force to get resultant force
                x[i][j] = x[i][j] + repulsive_force_magnitude * np.cos(angle_obstacle)
                y[i][j] = y[i][j] + repulsive_force_magnitude * np.sin(angle_obstacle)
                
                ## if the resultant force is 0, then the agent is in local minima and it cannot overcome it, hence care should be taken to avoid this.


## Plotting the path using streamplot function and extracting the waypoints
fig1, ax1 = plt.subplots() 
fig2, ax2 = plt.subplots()     
coordinates = []
for streamline in ax1.streamplot(map_x, map_y, x, y, start_points=[initial]).lines.get_paths():
    coords = streamline.vertices
    coordinates.append(coords*10)
coords_array = np.array(coordinates).astype(int)
for obstacle in obstacles:
    ax1.add_patch(plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='gray', alpha=0.3))
    ax2.add_patch(plt.Circle((obstacle[0]*10, obstacle[1]*10), obstacle[2]*10, color='gray', alpha=0.3))
ax1.set_title("Potential Force-Field Plot 1")
ax1.set_xlim([-300, 300])
ax1.set_ylim([-150, 150])
ax2.set_title("Potential Force-Field Plot 2")
ax1.set_aspect('equal')
ax2.set_xlim([-3000, 3000])
ax2.set_ylim([-1500, 1500])
ax2.set_aspect('equal')
x = coords_array[:,:,0].flatten()
y = coords_array[:,:,1].flatten()
ax2.plot(x, y, color='blue', linestyle='-')
plt.show()