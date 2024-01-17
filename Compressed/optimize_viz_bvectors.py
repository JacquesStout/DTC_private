import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DTC.diff_handlers.bvec_handler import read_bvals
from scipy.optimize import minimize
from itertools import combinations, permutations

# Function to calculate the angular distance between two vectors on a unit sphere
def angular_distance(v1, v2):
    return np.arccos(np.dot(v1, v2))

# Objective function to minimize the sum of angular distances between selected directions
def objective_function(selected_indices, directions):
    selected_indices = selected_indices.astype('int')
    selected_vectors = directions[selected_indices]
    total_distance = 0

    for i in range(len(selected_vectors)):
        for j in range(i + 1, len(selected_vectors)):
            total_distance += angular_distance(selected_vectors[i], selected_vectors[j])

    return total_distance


# Specify the center point
center = np.array([0, 0, 0])

bval_path = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/temp/20220905_14_checked.bval'
bvec_path = '/Volumes/dusom_mousebrains/All_Staff/Data/CS/MouseMRI_Duke_results/20220905_14/11/temp/20220905_14_checked.bvec'
bvalues,bvectors = read_bvals(bval_path,bvec_path, sftp=None)
threshold = 300
b0_mask = bvalues < threshold
dwi_mask = bvalues > threshold


cut_number = 21

bvectors = bvectors.T[dwi_mask][:]

bvectors_list = []
for i in np.arange(np.shape(bvectors)[0]):
#for i in np.arange(21):
    if not np.linalg.norm(np.array(bvectors[i]))<0.5:
        bvectors_list.append(np.array(bvectors[i]))

#np.random.seed(42)
#all_directions = np.random.rand(40, 3)  # 40 random 3D directions

optimize = True

full_sphere = True

if optimize:

    """
    First attempt
    result = minimize(objective_function, x0=np.arange(5,26), args=(bvectors,),
                      bounds=[(0, 39)] * 20, method='L-BFGS-B')
    selected_indices = result.x.astype(int)
    selected_vectors = all_directions[selected_indices]
    """

    """
    Second attempt
    selected_indices = []

    all_distances_mat = np.zeros([np.shape(bvectors)[0],np.shape(bvectors)[0]])
    for i in np.arange(np.shape(bvectors)[0]):
        for j in np.arange(np.shape(bvectors)[0]):
            if i==j:
                all_distances_mat[i,j] = 0
            else:
                all_distances_mat[i,j] = angular_distance(bvectors[i], bvectors[j])
                all_distances_mat[j,i] = all_distances_mat[i,j]

    for _ in range(cut_number):
        remaining_indices = list(set(range(np.shape(bvectors)[0])) - set(selected_indices))
        max_distance = 0
        selected_index = None

        for index in remaining_indices:
            total_distance = sum(all_distances_mat[index, remaining_indices])
            if total_distance > max_distance:
                max_distance = total_distance
                selected_index = index

        selected_indices.append(selected_index)

    selected_indices = sorted(selected_indices)
    """

    selected_indices = [0]
    selected_vectors = [bvectors_list[selected_indices[0]]]
    for _ in range(cut_number-1):
        remaining_vectors = [v for v in bvectors_list if list(v) not in np.array(selected_vectors)]
        max_min_distance = 0
        selected_vector = None

        for i,vector in enumerate(bvectors_list):
            if list(vector) not in np.array(remaining_vectors):
                continue
            min_distance = min(angular_distance(vector, v) for v in selected_vectors) if selected_vectors else 0
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                selected_vector = vector
                selected_indice = i

        selected_vectors.append(selected_vector)
        selected_indices.append(selected_indice)


else:
    selected_indices = np.arange(np.shape(bvectors)[0])[:cut_number]


visualization = True
if visualization:
    firstrun = False

    if firstrun:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the center point
        ax.scatter(*center, color='red', label='Center')

        # Plot each vector
        for vector in bvectors[selected_indices]:
            ax.quiver(*center, *vector, color=np.random.rand(3,), label='Vector')

        # Set labels and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('bvectors in 3D Space')

        # Display the plot
        plt.legend()
        plt.show()

    secondrun=True
    if secondrun:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the center point
        ax.scatter(*center, color='red', label='Center')

        # Plot each vector
        bvectors_list_selected = [bvectors_list[indice] for indice in selected_indices]

        if full_sphere:
            bvectors_list_selected = bvectors_list_selected + [-bvector for bvector in bvectors_list_selected]

        for vector in bvectors_list_selected:
            ax.quiver(*center, *vector, color=np.random.rand(3, ), label='Vector')

        # Plot a sphere of radius 1
        u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, color='c', alpha=0.3, label='Sphere')

        # Plot endpoints of bvectors_list on the sphere's surface
        for vector in bvectors_list_selected:
            endpoint = center + vector
            ax.scatter(*endpoint, color='blue', s=50, label='Endpoint')

        # Set labels and title
        #ax.set_xlabel('X-axis')
        #ax.set_ylabel('Y-axis')
        #ax.set_zlabel('Z-axis')
        ax.set_title('Vectors and Sphere in 3D Space')

        # Display the plot
        #plt.legend()
        plt.show()