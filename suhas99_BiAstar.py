'''
This code file is part of submission for ENPM661 Final Exam.

Bi-Directional A* algorithm for path planning is implemented in the code.

The map ranges from -3000mm to 3000mm in x axis and -1500mm to 1500mm in the y axis with the origin at the centre.

The initial point is at (-3000,0) and the goal point is at (3000,0)
'''

import pygame
import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
import time

## map is initialized as a numpy array with inf values
map1=np.full((601, 301), np.inf)
map2=np.full((601, 301), np.inf)
map_plot = np.full((6001, 3001), np.inf)

## Clearance space around obstacles
offset = 10

## Using semi algebraic equations to define obstacles and offset space
for x in range(601):
    for y in range(301):
        circle1 = (((x-112 )**2)+((y-242.5)**2)<=(40**2))
        circle2 = (((x-263)**2)+((y-90)**2)<=(70**2))
        circle3 = (((x-445)**2)+((y-220)**2)<=(37.5**2))
        circle1_offset = (((x-112)**2)+((y-242)**2)<=(40+offset)**2)
        circle2_offset = (((x-263)**2)+((y-90)**2)<=(70+offset)**2)
        circle3_offset = (((x-445)**2)+((y-220)**2)<=(37.5+offset)**2)
        if(circle1_offset or circle2_offset or circle3_offset):
            map1[x,y] = -2
            map2[x,y] = -2
        if(circle1 or circle2 or circle3):
            map1[x,y] = -1
            map2[x,y] = -1

## Offset for converting from map coordinate values to plot coordinate values
x_offset = 300
y_offset = 150

## Defining the initial point
initial_x = -300 + x_offset
initial_y = 0 + y_offset

## Defining the goal point
goal_x = 300 + x_offset
goal_y = 0 + y_offset

## Weight considered for computing cost to go
weight=1

''' Forward A*'''

## Goal node is the actual goal node
goal_node1 = (goal_x,goal_y)

## C2G is computed by taking goal node as reference
def get_c2g1(x,y):
    dist = math.sqrt(((goal_x-x)**2)+((goal_y-y)**2))
    return dist*weight

## Getting c2g from initial point
initial_c2g1 = get_c2g1(initial_x,initial_y)

## Initial node is the actual starting point
initial_node1 = [(initial_x , initial_y , 0 , initial_c2g1) , [(0,0)]]

initial_total_cost1 = initial_c2g1 + 0


''' Backward A*'''

## Goal node is the strating node
goal_node2 = (initial_x,initial_y)

## C2G is computed by taking initial node as reference
def get_c2g2(x,y):
    dist = math.sqrt(((initial_x-x)**2)+((initial_y-y)**2))
    return dist*weight

## Getting c2g from goal point
initial_c2g2 = get_c2g2(goal_x,goal_y)

## Initial node is the goal point
initial_node2 = [(goal_x , goal_y , 0 , initial_c2g2) , [(0,0)]]

initial_total_cost2 = initial_c2g2 + 0


## Defining the 8 action sets

## Moving up,down,right or left will cost 1 and moving diagonally will cost sqrt(2)

def top_left(node,type):
    (x,y,c2c,_)=node[0]
    path = node[1].copy()
    path.append((x, y))
    new_x = x -1
    new_y = y + 1
    new_c2c = c2c + np.sqrt(2)
    if type==1:
        new_c2g = get_c2g1(new_x,new_y)
    elif type==2:
        new_c2g = get_c2g2(new_x,new_y)
    new_node=[(new_x,new_y,new_c2c,new_c2g),path]

    return new_node

def top_right(node,type):
    (x,y,c2c,_)=node[0]
    path = node[1].copy()
    path.append((x, y))
    new_x = x + 1
    new_y = y + 1
    new_c2c = c2c + np.sqrt(2)
    if type==1:
        new_c2g = get_c2g1(new_x,new_y)
    elif type==2:
        new_c2g = get_c2g2(new_x,new_y)
    new_node=[(new_x,new_y,new_c2c,new_c2g),path]

    return new_node

def bottom_left(node,type):
    (x,y,c2c,_)=node[0]
    path = node[1].copy()
    path.append((x, y))
    new_x = x -1
    new_y = y - 1
    new_c2c = c2c + np.sqrt(2)
    if type==1:
        new_c2g = get_c2g1(new_x,new_y)
    elif type==2:
        new_c2g = get_c2g2(new_x,new_y)
    new_node=[(new_x,new_y,new_c2c,new_c2g),path]

    return new_node

def bottom_right(node,type):
    (x,y,c2c,_)=node[0]
    path = node[1].copy()
    path.append((x, y))
    new_x = x + 1
    new_y = y - 1
    new_c2c = c2c + np.sqrt(2)
    if type==1:
        new_c2g = get_c2g1(new_x,new_y)
    elif type==2:
        new_c2g = get_c2g2(new_x,new_y)
    new_node=[(new_x,new_y,new_c2c,new_c2g),path]

    return new_node


def bottom(node,type):
    (x,y,c2c,_)=node[0]
    path = node[1].copy()
    path.append((x, y))
    new_x = x
    new_y = y - 1
    new_c2c = c2c + 1
    if type==1:
        new_c2g = get_c2g1(new_x,new_y)
    elif type==2:
        new_c2g = get_c2g2(new_x,new_y)
    new_node=[(new_x,new_y,new_c2c,new_c2g),path]

    return new_node

def top(node,type):
    (x,y,c2c,_)=node[0]
    path = node[1].copy()
    path.append((x, y))
    new_x = x
    new_y = y + 1
    new_c2c = c2c + 1
    if type==1:
        new_c2g = get_c2g1(new_x,new_y)
    elif type==2:
        new_c2g = get_c2g2(new_x,new_y)
    new_node=[(new_x,new_y,new_c2c,new_c2g),path]

    return new_node

def right(node,type):
    (x,y,c2c,_)=node[0]
    path = node[1].copy()
    path.append((x, y))
    new_x = x + 1
    new_y = y
    new_c2c = c2c + 1
    if type==1:
        new_c2g = get_c2g1(new_x,new_y)
    elif type==2:
        new_c2g = get_c2g2(new_x,new_y)
    new_node=[(new_x,new_y,new_c2c,new_c2g),path]

    return new_node

def left(node,type):
    (x,y,c2c,_)=node[0]
    path = node[1].copy()
    path.append((x, y))
    new_x = x - 1
    new_y = y
    new_c2c = c2c + 1
    if type==1:
        new_c2g = get_c2g1(new_x,new_y)
    elif type==2:
        new_c2g = get_c2g2(new_x,new_y)
    new_node=[(new_x,new_y,new_c2c,new_c2g),path]

    return new_node

##  Creating open list, closed list and tracking array for both forward and backward search
open_list1 = []
heapq.heapify(open_list1)
heapq.heappush(open_list1, (initial_total_cost1, initial_node1))
open_list2 = []
heapq.heapify(open_list2)
heapq.heappush(open_list2, (initial_total_cost2, initial_node2))
closed_list1 = []
closed_list2 = []
tracking_array_closed1 = map1.copy()
tracking_array_closed2 = map2.copy()
searching = True
possible_actions = [top_left,top_right,bottom_left,bottom_left,right,left,top,bottom]


while searching:
    ## The node with lowest total cost is popped from the open list
    (total_cost1,node1) = heapq.heappop(open_list1)
    (total_cost2,node2) = heapq.heappop(open_list2)
    (x1,y1,c2c1,c2g1)=node1[0]
    (x2,y2,c2c2,c2g2)=node2[0]
    current_path1=node1[1]
    current_path2=node2[1]

    ## The parent - child values are appended for plotting
    (last_x1,last_y1) = current_path1[-1]
    (last_x2,last_y2) = current_path2[-1]
    tracking_node1=[x1,y1,last_x1,last_y1]
    tracking_node2=[x2,y2,last_x2,last_y2]
    closed_list1.append(tracking_node1)
    closed_list2.append(tracking_node2)

    ## The popped nodes are added to closed list. Here the tracking node value is changed to indicate that
    tracking_array_closed1[int(x1),int(y1)] = 2
    tracking_array_closed2[int(x2),int(y2)] = 2

    ## If the 2 searches meet, they will have a common point, this is the termination condition
    if tracking_array_closed1[int(x2),int(y2)]==2 and tracking_array_closed2[int(x1),int(y1)] == 2:
        print("Solution reached")
        searching = False
        final_node1 = node1
        final_node2 = node2
    else:
        ## The action set is applied to the popped node (forward)
        for move_function in possible_actions:
            new_node = move_function(node1,1)
            (new_x, new_y,new_c2c,new_c2g) = new_node[0]

            ## the validity of new node is checked
            if new_x>600 or new_x<0 or new_y>300 or new_y<0:
                continue

            ## total cost for the new node is computed
            new_total_cost = new_c2c + new_c2g

            ## A test node is created for checking with open list
            test = (int(new_x), int(new_y))

            # if the new node is in obstacle or offset space, skip the node
            if map1[int(new_x),int(new_y)] < 0:
                continue

            # if the new node is in closed, skip the node. This is checked using a 3d array
            if tracking_array_closed1[int(new_x),int(new_y)]==2:
                continue

            # if the node is in open list, check if the new c2c < old c2c, if so, replace the node.
            if map1[int(new_x),int(new_y)]==1:
                for i, (_, open_node) in enumerate(open_list1):
                    (open_x,open_y,open_c2c,_) = open_node[0]
                    open_test = (open_x,open_y)
                    if open_test==test:
                        if new_c2c < open_c2c:
                            open_list1[i] = (new_total_cost, new_node)
                            break
            else:
                # if the open node is nither of the above, add it to the open list
                map1[int(new_x),int(new_y)]=1
                heapq.heappush(open_list1, (new_total_cost, new_node))

        ## The action set is applied to the popped node (backward)
        for move_function in possible_actions:
            new_node = move_function(node2,2)
            (new_x, new_y,new_c2c,new_c2g) = new_node[0]

            ## the validity of new node is checked
            if new_x>600 or new_x<0 or new_y>300 or new_y<0:
                continue

            ## total cost for the new node is computed
            new_total_cost = new_c2c + new_c2g

            ## A test node is created for checking with open list
            test = (int(new_x), int(new_y))

            # if the new node is in obstacle or offset space, skip the node
            if map2[int(new_x),int(new_y)] < 0:
                continue
            # if the new node is in closed, skip the node. This is checked using a 3d array
            if tracking_array_closed2[int(new_x),int(new_y)]==2:
                continue
            # if the node is in open list, check if the new c2c < old c2c, if so, replace the node.
            if map2[int(new_x),int(new_y)]==1:
                for i, (_, open_node) in enumerate(open_list2):
                    (open_x,open_y,open_c2c,_) = open_node[0]
                    open_test = (open_x,open_y)
                    if open_test==test:
                        if new_c2c < open_c2c:
                            open_list2[i] = (new_total_cost, new_node)
                            break
            else:
                # if the open node is nither of the above, add it to the open list
                map2[int(new_x),int(new_y)]=1
                heapq.heappush(open_list2, (new_total_cost, new_node))

        ## If both open lists are empty, then it is declared that no solution is found
        if not open_list1 and not open_list2:
            print("No solution found")

## Pygame animation
## Forward search is denoted in pink
## Backward search is denoted in green
## Final path is denoted in black
## Obstacle is denoted in red
## Clearance is denoted in yellow
pygame.init()
screen = pygame.display.set_mode((600, 300))
running = True
while running:
    # Indexing for skipping the first value, a dummy value which we added earlier
    n1=0
    n2=0
    m1=0
    m2=0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ## Plotting obstacles
    for y in range(map1.shape[1]):
        for x in range(map1.shape[0]):
            if map1[x, y] >=0 :
                pygame.draw.rect(screen, (255, 255, 255), (x , 300-y , 1, 1))
            elif map1[x, y] == -2:
                pygame.draw.rect(screen, (255, 255, 0), (x , 300-y , 1, 1))
            elif map1[x, y] == -1:
                pygame.draw.rect(screen, (255, 0, 0), (x, 300-y , 1, 1))

    ## Plotiing initial and goal nodes
    pygame.draw.rect(screen, (0, 0, 0), (goal_node1[0] , 300-goal_node1[1] , 1, 1))
    pygame.draw.rect(screen, (0, 0, 0), (initial_node1[0][0] , 300-initial_node1[0][1] , 1, 1))

    ## Plotiing forward search node exploration
    for i in closed_list1:
        if n1==0:
            n1=n1+1
            continue
        (x_current,y_current,x_previous,y_previous) = i
        pygame.draw.line(screen, (255,150,140), (x_previous, 300-y_previous), (x_current,300-y_current),1)
        pygame.display.update()

    ## Plotiing backward search node exploration
    for i in closed_list2:
        if n2==0:
            n2=n2+1
            continue
        (x_current,y_current,x_previous,y_previous) = i
        pygame.draw.line(screen, (0,150,140), (x_previous, 300-y_previous), (x_current,300-y_current),1)
        pygame.display.update()

    ## Extracting path from forward search and plotting
    path1=final_node1[1]
    for i in range(len(path1)-1):
        if m1==0:
            m1=m1+1
            continue
        (x1,y1)=path1[i]
        (x2,y2)=path1[i+1]
        pygame.draw.line(screen, (0,0,0), (x1,300-y1),(x2,300-y2),2)
    x3,y3=path1[-1]
    x4, y4, _, _ = final_node1[0]
    pygame.draw.line(screen, (0,0,0), (x3,300-y3),(x4,300-y4),2)

    ## Extracting path from backward search and plotting
    path2=final_node2[1]
    for i in range(len(path2)-1):
        if m2==0:
            m2=m2+1
            continue
        (x5,y5)=path2[i]
        (x6,y6)=path2[i+1]
        pygame.draw.line(screen, (0,0,0), (x5,300-y5),(x6,300-y6),2)
    x7,y7=path2[-1]
    x8, y8, _, _ = final_node2[0]
    pygame.draw.line(screen, (0,0,0), (x7,300-y7),(x8,300-y8),2)
    ## Combining the 2 paths
    pygame.draw.line(screen, (0,0,0), (x4,300-y4),(x8,300-y8),2)
    pygame.display.update()
    time.sleep(10)
    pygame.image.save(screen, 'Figure_3.png')
    running=False
pygame.quit()

## Matplotlib plotting
## The obstacles are defined in the original scale
for x in range(6001):
    for y in range(3001):
        circle1 = (((x-1120 )**2)+((y-2425)**2)<=(400**2))
        circle2 = (((x-2630)**2)+((y-900)**2)<=(700**2))
        circle3 = (((x-4450)**2)+((y-2200)**2)<=(375**2))
        circle1_offset = (((x-1120)**2)+((y-2425)**2)<=(400+(offset*10))**2)
        circle2_offset = (((x-2630)**2)+((y-900)**2)<=(700+(offset*10))**2)
        circle3_offset = (((x-4450)**2)+((y-2200)**2)<=(375+(offset*10))**2)
        if(circle1_offset or circle2_offset or circle3_offset):
            map_plot[x,y] = -2
        if(circle1 or circle2 or circle3):
            map_plot[x,y] = -1

## The obstacles are plotted
plt.imshow(map_plot.T, cmap='jet')

## The path values obtained are sclaed up to original scale
scale=10

## The node exploration is plotted
for x_current, y_current, x_previous, y_previous in closed_list1[1:]:
    plt.plot([x_previous*scale, x_current*scale], [y_previous*scale, y_current*scale], color='lightgreen')
for x_current, y_current, x_previous, y_previous in closed_list2[1:]:
    plt.plot([x_previous*scale, x_current*scale], [y_previous*scale, y_current*scale], color='pink')

# Paths are plotted
path1 = final_node1[1]
for i in range(len(path1)-2):
    x1, y1 = path1[i+1]
    x2, y2 = path1[i+2]
    plt.plot([x1*scale, x2*scale], [y1*scale, y2*scale], color='black')
x3, y3 = path1[-1]
x4, y4, _, _ = final_node1[0]
plt.plot([x3*scale, x4*scale], [y3*scale, y4*scale], color='black')
path2 = final_node2[1]
for i in range(len(path2)-2):
    x5, y5 = path2[i+1]
    x6, y6 = path2[i+2]
    plt.plot([x5*scale, x6*scale], [y5*scale, y6*scale], color='black')
x7, y7 = path2[-1]
x8, y8, _, _ = final_node2[0]
plt.plot([x7*scale, x8*scale], [y7*scale, y8*scale], color='black')

plt.plot([x4*scale, x8*scale], [y4*scale, y8*scale], color='black')

plt.gca().invert_yaxis()
plt.savefig('Figure_4.png')
plt.show()
