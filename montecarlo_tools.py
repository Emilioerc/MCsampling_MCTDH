import numpy as np
import os
import subprocess
import logging
from typing import List, Optional
from sklearn.cluster import KMeans
import multiprocessing as mp
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_plot_data(file_path):
    """
    Extracts data between 'plot' and 'e' lines in the input file, stores each block as numerical data,
    and counts the number of lines in each block.

    Parameters:
        file_path (str): Path to the file containing the input data.
    
    Returns:
        tuple: 
            - list of numpy arrays where each array contains the data (as floats) from one 'plot' block.
            - list of integers where each element is the number of lines for the corresponding block.
    """
    all_data_blocks = []
    line_counts = []
    current_block = []
    current_line_count = 0
    inside_block = False

    try:
        with open(file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()

                if 'plot' in stripped_line:
                    # Start of a new block
                    if current_block:
                        # Save the previous block
                        all_data_blocks.append(np.array(current_block))
                        line_counts.append(current_line_count)
                    current_block = []
                    current_line_count = 0
                    inside_block = True
                elif stripped_line == 'e':
                    # End of current block
                    inside_block = False
                    if current_block:
                        all_data_blocks.append(np.array(current_block))
                        line_counts.append(current_line_count)
                    current_block = []
                    current_line_count = 0
                elif inside_block:
                    # Collect data
                    try:
                        numerical_values = list(map(float, stripped_line.split()))
                        current_block.append(numerical_values)
                        current_line_count += 1
                    except ValueError:
                        logger.warning(f"Non-numerical line encountered: '{stripped_line}'")
                        continue
            # After the loop, check if there's any data in current_block
            if current_block:
                all_data_blocks.append(np.array(current_block))
                line_counts.append(current_line_count)
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found.")
        return [], []

    return all_data_blocks, line_counts

def compute_cdf(density_values):
    """Convert the density array into a CDF array."""
    total = np.sum(density_values)
    if total == 0:
        raise ValueError("Total density is zero, cannot compute CDF.")
    cdf = np.cumsum(density_values) / total
    return cdf

def inverse_transform_sampling(cdf, grid_points, num_samples):
    """Perform inverse transform sampling using the CDF and return indices and corresponding grid points."""
    random_samples = np.random.rand(num_samples)
    sampled_indices = np.searchsorted(cdf, random_samples, side='right')
    sampled_indices = np.clip(sampled_indices, 0, len(grid_points) - 1)
    sampled_grid_points = grid_points[sampled_indices]
    return sampled_grid_points, sampled_indices

def sample_n_dofs(full_1dden_data, num_samples, idx_time):
    """
    Perform sampling for n degrees of freedom using densities and grid points.

    Parameters:
        full_1dden_data: list containing density and grid points for each DoF
        num_samples: number of samples to draw for each DoF
        idx_time: index representing the time step

    Returns:
        sampled_grid_points: A 2D numpy array where each row corresponds to sampled grid points.
        sampled_indices: A 2D numpy array where each row corresponds to the indices in the full configurational space.
    """
    n_dofs = len(full_1dden_data)  # Number of degrees of freedom (DoFs)
    sampled_grid_points_data = []
    sampled_indices_data = []

    for dof in range(n_dofs):
        # Extract densities and grid points for this DoF
        densities = full_1dden_data[dof][idx_time][:, 1]
        grid_points = full_1dden_data[dof][idx_time][:, 0]

        try:
            # Compute the CDF for the current DoF
            cdf = compute_cdf(densities)
        except ValueError as e:
            logger.error(f"Error computing CDF for DoF {dof}: {e}")
            continue

        # Perform sampling for the current DoF and get both grid points and indices
        sampled_grid_points, sampled_indices = inverse_transform_sampling(cdf, grid_points, num_samples)

        # Store the sampled grid points and indices for this DoF
        sampled_grid_points_data.append(sampled_grid_points)
        sampled_indices_data.append(sampled_indices)

    # Combine the samples for all DoFs into a full configurational space
    sampled_grid_points = np.column_stack(sampled_grid_points_data)
    sampled_indices = np.column_stack(sampled_indices_data)

    return sampled_grid_points, sampled_indices

def remove_duplicates(samples, tolerance=1e-5):
    """
    Remove duplicate points from the sampled data based on a tolerance.

    Parameters:
        samples: numpy array containing the sampled points.
        tolerance: Tolerance level to consider points as duplicates.

    Returns:
        unique_samples: A numpy array of unique sampled points.
    """
    # Round points to tolerance level and remove duplicates
    decimals = max(int(-np.log10(tolerance)), 0)
    rounded_samples = np.round(samples, decimals=decimals)
    _, unique_indices = np.unique(rounded_samples, axis=0, return_index=True)
    unique_samples = samples[sorted(unique_indices)]

    return unique_samples

def run_fortran_executable(executable_path, flags, grid_indices):
    """
    Runs the Fortran executable with specified flags and provides grid indices as input.

    Parameters:
        executable_path (str): Path to the Fortran executable.
        flags (list): List of command-line flags (e.g., ['-w', '-ort']).
        grid_indices (list or array-like): List of integers representing grid indices (e.g., [1, 2, 3]).

    Returns:
        psi_values (list of complex): List of complex psi values for each state.
    """
    # Check if the executable exists
    if not os.path.isfile(executable_path):
        logger.error(f"Executable '{executable_path}' not found.")
        return None

    # Build the command
    cmd = [executable_path] + flags
    logger.debug(f"Executing command: {' '.join(cmd)} with grid indices: {grid_indices}")

    # Prepare the input string
    input_str = ' '.join(map(str, grid_indices)) + '\n'
    logger.debug(f"Input to Fortran executable: '{input_str.strip()}'")

    try:
        # Execute the Fortran program
        result = subprocess.run(
            cmd,
            input=input_str,
            capture_output=True,
            text=True,
            check=True
            # timeout=60  # Optional: Add timeout if needed
        )

        logger.debug(f"Return Code: {result.returncode}")
        logger.debug(f"Standard Output:\n{result.stdout}")
        logger.debug(f"Standard Error:\n{result.stderr}")

        # Parse the output to extract psi values
        psi_values = parse_psi_output(result.stdout)
        return psi_values

    except subprocess.CalledProcessError as e:
        # Handle errors in execution
        logger.error("An error occurred while running the Fortran executable.")
        logger.error(f"Return Code: {e.returncode}")
        logger.error(f"Standard Output:\n{e.stdout}")
        logger.error(f"Standard Error:\n{e.stderr}")
        return None

def parse_psi_output(output_str):
    """
    Parses the output string from the Fortran executable to extract psi values.

    Parameters:
        output_str (str): The output string from the Fortran executable.

    Returns:
        psi_values (list of complex): List of complex psi values for each state.
    """
    psi_values = []
    # Pattern to match the psi value lines
    pattern = r"The value of psi for state.*\n\s*([\d\.\-\+eE]+)\s+([\d\.\-\+eE]+)"
    matches = re.findall(pattern, output_str)
    for real_part_str, imag_part_str in matches:
        try:
            real_part = float(real_part_str)
            imag_part = float(imag_part_str)
            psi_values.append(complex(real_part, imag_part))
        except ValueError:
            logger.error(f"Error parsing psi values: '{real_part_str} {imag_part_str}'")
    return psi_values

def cluster_sampling(indices, grid_points, n_clusters):
    """
    Apply K-means clustering to the indices and return matching grid points and indices.

    Parameters:
        indices: numpy array containing the sampled indices (for clustering).
        grid_points: numpy array containing the corresponding grid points.
        n_clusters: Number of clusters to create for representative sampling.

    Returns:
        representative_grid_points: A numpy array of representative grid points from each cluster.
        representative_indices: A numpy array of representative indices from each cluster (adjusted for Fortran indexing).
    """
    n_samples = indices.shape[0]
    if n_samples < n_clusters:
        logger.warning(f"Number of clusters ({n_clusters}) is greater than available samples ({n_samples}). Reducing clusters to {n_samples}.")
        n_clusters = n_samples

    # Cluster based on indices
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(indices)

    # Get labels for each sample
    labels = kmeans.labels_

    # For each cluster, pick the sample closest to the cluster center
    representative_indices = []
    representative_grid_points = []

    for cluster_id in range(n_clusters):
        # Get indices of samples in this cluster
        cluster_sample_indices = np.where(labels == cluster_id)[0]

        # Get the samples in this cluster
        cluster_indices = indices[cluster_sample_indices]
        cluster_grid_points = grid_points[cluster_sample_indices]

        # Get the cluster center
        cluster_center = kmeans.cluster_centers_[cluster_id]

        # Compute distances to the cluster center
        distances = np.linalg.norm(cluster_indices - cluster_center, axis=1)

        # Find the sample closest to the cluster center
        closest_sample_idx = cluster_sample_indices[np.argmin(distances)]

        # Append the representative sample
        representative_indices.append(indices[closest_sample_idx])
        representative_grid_points.append(grid_points[closest_sample_idx])

    # Convert to numpy arrays
    representative_indices = np.array(representative_indices)
    representative_grid_points = np.array(representative_grid_points)

    # Adjust indices for Fortran indexing (starting from 1)
    representative_indices += 1

    return representative_grid_points, representative_indices

def save_samples_and_indices(grid_samples, index_samples, filename_prefix='mcmc'):
    """
    Save both grid points and indices to separate NumPy binary files.

    Parameters:
        grid_samples (np.ndarray): Array of sampled grid points.
        index_samples (np.ndarray): Array of sampled indices.
        filename_prefix (str): Prefix for the saved files.
    """
    grid_filename = f'{filename_prefix}_grid_samples.npy'
    index_filename = f'{filename_prefix}_index_samples.npy'
    
    np.save(grid_filename, grid_samples)
    np.save(index_filename, index_samples)
    
    logger.info(f"Grid samples saved to '{grid_filename}'")
    logger.info(f"Index samples saved to '{index_filename}'")

def run_and_collect(executable_path: str, flags: List[str], grid_indices: List[int]) -> Optional[float]:
    """
    Runs the Fortran executable and collects the squared absolute psi values.
    
    Parameters:
        executable_path (str): Path to the Fortran executable.
        flags (List[str]): Command-line flags for the executable.
        grid_indices (List[int]): Grid indices to pass as input.
    
    Returns:
        Optional[float]: Sum of squared absolute psi values or None if execution fails.
    """
    psi_values = run_fortran_executable(executable_path, flags, grid_indices)
    if psi_values is not None:
        # Calculate and return the sum of probability densities
        return np.sum(np.abs(psi_values) ** 2)
    else:
        return None
    
def get_neighbors_single_dimension(index, shape, boundary='reflect', max_step=1):
    """
    Get neighboring indices by updating only one randomly selected dimension.
    """
    neighbors = []
    index = np.array(index)
    dims = len(index)
    for d in range(dims):
        for step in range(-max_step, max_step + 1):
            if step == 0:
                continue
            neighbor = index.copy()
            neighbor[d] += step
            # Apply boundary conditions
            if neighbor[d] < 0 or neighbor[d] >= shape[d]:
                if boundary == 'periodic':
                    neighbor[d] %= shape[d]
                elif boundary == 'reflect':
                    neighbor[d] = max(0, min(neighbor[d], shape[d] - 1))
                else:
                    raise ValueError(f"Unknown boundary condition: '{boundary}'")
            neighbors.append(tuple(neighbor))
    return neighbors

def propose_random_neighbor(index, shape, boundary='reflect', max_step=1):
    """
    Propose a neighbor by randomly updating a subset of dimensions.
    """
    index = np.array(index)
    dims = len(index)
    # Decide how many dimensions to update (e.g., 1 or 2)
    num_dims_to_update = np.random.randint(1, min(3, dims + 1))  # Update 1 or 2 dimensions
    dimensions_to_update = np.random.choice(dims, num_dims_to_update, replace=False)
    neighbor = index.copy()
    for d in dimensions_to_update:
        step = np.random.randint(-max_step, max_step + 1)
        neighbor[d] += step
        # Apply boundary conditions
        if neighbor[d] < 0 or neighbor[d] >= shape[d]:
            if boundary == 'periodic':
                neighbor[d] %= shape[d]
            elif boundary == 'reflect':
                neighbor[d] = max(0, min(neighbor[d], shape[d] - 1))
            else:
                raise ValueError(f"Unknown boundary condition: '{boundary}'")
    return tuple(neighbor)


def get_neighbors(index, shape, boundary='reflect', max_step=1):
    """
    Get neighboring indices on the grid with boundary conditions.

    Parameters:
    - index: Tuple representing the current grid index.
    - shape: Tuple representing the grid shape.
    - boundary: 'reflect' or 'periodic'.
    - max_step: Maximum step size in any dimension.

    Returns:
    - neighbors: List of tuples representing valid neighboring indices.
    """
    neighbors = []
    index = np.array(index)
    dims = len(index)
    deltas = range(-max_step, max_step + 1)

    # Generate all possible delta combinations
    delta_grids = np.meshgrid(*([deltas] * dims))
    delta_combinations = np.array(delta_grids).T.reshape(-1, dims)

    for delta in delta_combinations:
        if np.all(delta == 0):
            continue  # Skip the current position

        neighbor = index + delta
        valid = True

        for d in range(dims):
            if neighbor[d] < 0 or neighbor[d] >= shape[d]:
                if boundary == 'periodic':
                    neighbor[d] %= shape[d]
                elif boundary == 'reflect':
                    neighbor[d] = max(0, min(neighbor[d], shape[d] - 1))
                else:
                    logger.error(f"Unknown boundary condition: '{boundary}'")
                    raise ValueError(f"Unknown boundary condition: '{boundary}'")
        if valid:
            neighbors.append(tuple(neighbor))

    return neighbors

def propose_gaussian_neighbor(index, shape, sigma=1.0):
    """
    Propose a neighbor using a Gaussian distribution centered at the current index.
    """
    index = np.array(index)
    dims = len(index)
    neighbor = index + np.random.normal(0, sigma, size=dims).astype(int)
    # Apply boundary conditions
    for d in range(dims):
        if neighbor[d] < 0 or neighbor[d] >= shape[d]:
            neighbor[d] = max(0, min(neighbor[d], shape[d] - 1))
    return tuple(neighbor)


def metropolis_worker(args):
    """Worker function for multiprocessing."""
    (initial_index, initial_prob, grid_shape, run_and_collect_fn, executable_path, flags, num_steps, boundary, shared_cache) = args
    current_index = initial_index
    current_prob = initial_prob
    samples = [(current_index, current_prob)]  # Store tuples of (position, density)
    cache = shared_cache

    for _ in range(num_steps):
        # Propose a neighbor
        # neighbors = propose_random_neighbor(current_index, grid_shape, boundary, max_step=1)
        neighbors = get_neighbors(current_index, grid_shape, boundary, max_step=2)
        if not neighbors:
            continue  # No valid neighbors
        proposed_index = neighbors[np.random.randint(len(neighbors))]

        # Check cache for density
        proposed_key = tuple(proposed_index)
        new_den = cache.get(proposed_key)
        if new_den is None:
            # Compute new density using run_and_collect
            new_den = run_and_collect_fn(executable_path, flags, list(proposed_index))
            if new_den is None:
                continue  # Skip if density could not be computed
            cache[proposed_key] = new_den  # Store in cache

        if new_den == 0 or current_prob == 0:
            continue  # Cannot compute acceptance, skip

        acceptance = new_den / current_prob
        if acceptance >= 1 or np.random.rand() < acceptance:
            current_index = proposed_index
            current_prob = new_den
            samples.append((current_index, current_prob))  # Store position and density
    return samples

def metropolis_parallel(initial_indices, initial_sp_psi, grid_shape, run_and_collect_fn, executable_path, flags, num_steps, boundary='reflect'):
    """
    Parallel Metropolis algorithm using multiprocessing.

    Parameters:
        initial_indices: List of initial indices for each walker.
        initial_sp_psi: List of initial probability densities for each walker.
        grid_shape: Shape of the grid.
        run_and_collect_fn: Function to compute probability densities.
        executable_path: Path to the Fortran executable.
        flags: Command-line flags for the executable.
        num_steps: Number of Metropolis steps.
        boundary: Boundary condition ('reflect' or 'periodic').

    Returns:
        List of samples from each worker.
    """
    N = len(initial_indices)  # Number of walkers

    # Shared cache for densities
    manager = mp.Manager()
    shared_cache = manager.dict()

    # Prepare arguments for each worker
    args_list = [
        (
            initial_indices[i],
            initial_sp_psi[i],
            grid_shape,
            run_and_collect_fn,
            executable_path,
            flags,
            num_steps,
            boundary,
            shared_cache
        )
        for i in range(N)
    ]

    # Use multiprocessing pool to run walkers in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(metropolis_worker, args_list)

    return results
