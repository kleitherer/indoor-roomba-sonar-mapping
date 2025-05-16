# The goal of this problem is to visualize the occupancy grid of a Roomba 
# at 30%, 60%, and 100% of its final trajectory

# The occupancy grid is a 8x8 grid, where each cell represents a 0.1 meter area.
# The roomba starts at center of the grid (4,4) 


import matplotlib.pyplot as plt
import numpy as np

#sonar_data = np.loadtxt('sonar_dataset.csv', delimiter=',', skiprows=1)
sonar_data = np.genfromtxt('sonar_dataset.csv', delimiter=',', skip_header=1, filling_values=np.nan)
# The sonar data is structured as follows:
# Column 0: X coordinate
# Column 1: Y coordinate
# Column 2: Heading angle
# Columns 3-14: Sonar measurements from 12 sensors
# Note that certain measurements are missing, which correspond to the cases where no
# target was within the maximum detection range or the signal was too weak or noisy

# Function to visualize the occupancy grid with detailed grid lines
def visualize_occupancy_grid(matrix, title="Occupancy Grid"):
    plt.imshow(matrix, cmap='gray_r', origin='lower', extent=[0, 8, 0, 8], vmin=0, vmax=1)
    plt.colorbar(label='Occupancy Probability')
    plt.title(title)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.xticks(np.arange(0, 8.1, 1))  # Major ticks every 1 meter
    plt.yticks(np.arange(0, 8.1, 1))  # Major ticks every 1 meter
    plt.minorticks_on()
    plt.gca().set_xticks(np.arange(0, 8.1, 0.1), minor=True)  # Minor ticks every 0.1 meter
    plt.gca().set_yticks(np.arange(0, 8.1, 0.1), minor=True)  # Minor ticks every 0.1 meter
    
    plt.show()

"""
Bresenham's Line Algorithm
Returns a list of grid cells between two points (x0, y0) and (x1, y1)

"""
def bresenham(x0, y0, x1, y1):
    # Convert to integers and scale to grid coordinates (0.1m resolution)
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    points = [] 

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1 
    err = dx - dy

    while True:
        points.append((x0, y0))  # Add the current cell to the list
        if x0 == x1 and y0 == y1:  # Stop when the end point is reached
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


# Function to update a specific cell in the matrix
# where p(mij|rk) is the probability of occupancy given the sensor reading rk
# and Lk-1 is the previous log-odd value
# L0 = 0 

def update_cell(matrix, row, col, new_value):
    """
    Updates the occupancy probability of a cell in the matrix using the Bayesian update rule.
    """
    if 0 <= row < len(matrix) and 0 <= col < len(matrix[0]):
        current_probability = matrix[row][col]
        current_logodds = np.log(current_probability / (1 - current_probability))
        
        measurement_logodds = np.log(new_value / (1 - new_value))
        updated_logodds = current_logodds + measurement_logodds
        
        updated_probability = 1 - (1 / (1 + np.exp(updated_logodds)))
        matrix[row][col] = updated_probability
    else:
        print(f"Error: Cell location ({row}, {col}) out of bounds.")


# define an 8x8m grid with each cell representing a 0.1m area
# the grid is initialized to 0.5 (50% occupancy)
y, x = 80, 80  
initial_probability = 0.5  
matrix = [[initial_probability for _ in range(x)] for _ in range(y)]


# iterate through the sonar data, for each row (which is a 1s time interval), 
# the robot should start at the center of the grid (4,4) and move to the next point
# we need to convert the coordinates from the sonar data to the grid coordinates


k = sonar_data.shape[0] # number of time steps 
position = np.zeros((2, k)) 
position[0] = sonar_data[:, 0] 
position[1] = sonar_data[:, 1] 

sonar_angles = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])

heading_angle = sonar_data[:, 2]


global_frame = np.zeros((12, k)) 

sonar_measurements = sonar_data[:, 3:15] # sonar measurements from 12 sensors

print("Estimated position start:", position[0, 0], position[1, 0])

for i in range(k-1):
    start_x = position[0][i] 
    start_y = position[1][i] 
    
    next_x = position[0][i+1]
    next_y = position[1][i+1]
    print("Start point:", start_x, start_y)
    print("End point:", next_x, next_y)
    # Plot trajectory line
    plt.plot([position[0][i], position[0][i+1]], [position[1][i], position[1][i+1]], 'r-', linewidth=2)


    # we need to convert the coordinates from the sonar data to the grid coordinates
    # if we are looking at sonar #1 (out of 12), its position/angle is 0 degrees from the heading
    # if we are looking at sonar #11 (out of 12), its position/angle is 330 degrees from the heading

    # so if we're at a specific position, we need to rotate the sonar measurements, based on its 

    for j in range(12):
        # get the angle of the sonar sensor
        sonar_angle = sonar_angles[j] + heading_angle[i]
        sonar_angle = np.deg2rad(sonar_angle)

        # take the local sonar measurement and convert it to global coordinates
        if not np.isnan(sonar_measurements[i, j]):
            sonar_x = start_x + np.cos(sonar_angle) * sonar_measurements[i, j]
            sonar_y = start_y + np.sin(sonar_angle) * sonar_measurements[i, j]
            grid_x = int(sonar_x * 10)
            grid_y = int(sonar_y * 10)

            # Mark the detected obstacle cell as occupied
            #update_cell(matrix, grid_y, grid_x, 0.8)
            start_grid_x = int(start_x * 10)
            start_grid_y = int(start_y * 10)

            free_cells = bresenham(start_grid_x, start_grid_y, grid_x, grid_y)
            for cell in free_cells[:-1]: 
                cell_x, cell_y = cell
                # update the occupancy grid to be free based on bayesian update rule
                update_cell(matrix, cell_y, cell_x, 0.2) 
            hit_x, hit_y = free_cells[-1]
            update_cell(matrix, hit_y, hit_x, 0.8)
            #update_cell(matrix, cell_y, cell_x, 0.8) # mark the last cell as occupied
        else:
            print(f"No sonar measurement at position {start_x,start_y} for sonar {j+1}")
        #     # if the measurement is NaN, then it means there is no target was within the maximum detection range
        #     sonar_x = start_x + np.cos(sonar_angle) * 6
        #     sonar_y = start_y + np.sin(sonar_angle) * 6
        #     grid_x = int(sonar_x * 10)
        #     grid_y = int(sonar_y * 10)

        #     notarget_cells = bresenham(start_grid_x, start_grid_y, grid_x, grid_y)
        #     for cell in notarget_cells: 
        #         cell_x, cell_y = cell
        #         # update the occupancy grid to be free based on bayesian update rule
        #         #update_cell(matrix, cell_y, cell_x, 0.5) 
        #         update_cell(matrix, cell_y, cell_x, 0.5) 

    # use Bresenham to get all the discrete grid cells between the start and end points
    # we will iterate through this entire list and update the occupancy grid to be free, 
    # except for the last cell, which will be occupied

    






visualize_occupancy_grid(matrix, title="100% Occupancy Grid with Trajectory")




