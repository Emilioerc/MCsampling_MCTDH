import numpy as np
import os
import subprocess
import logging
from typing import List, Optional, Union, Tuple, Any, Callable
from sklearn.cluster import KMeans
import multiprocessing as mp
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_plot_data(file_path: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    Extract data blocks from a file between lines containing 'plot' and 'e'.

    This function reads the specified file line by line, identifies blocks of data
    that begin with a line containing the substring 'plot' and end with a line containing
    the character 'e'. Each identified block is converted into a NumPy array of floats,
    and the number of lines in each block is recorded.

    Parameters
    ----------
    file_path : str
        The path to the input file containing the data blocks.

    Returns
    -------
    Tuple[List[np.ndarray], List[int]]
        A tuple containing two lists:
            - The first list contains NumPy arrays, each representing a data block.
            - The second list contains integers, each representing the number of lines
              in the corresponding data block.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.

    Notes
    -----
    - Lines containing 'plot' are considered the start of a new data block.
    - Lines containing only 'e' are considered the end of the current data block.
    - Non-numerical lines within a data block are skipped with a warning.
    """
    all_data_blocks: List[np.ndarray] = []
    line_counts: List[int] = []
    current_block: List[List[float]] = []
    current_line_count: int = 0
    inside_block: bool = False

    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                stripped_line = line.strip()

                if 'plot' in stripped_line:
                    # Start of a new block
                    if current_block:
                        # Save the previous block
                        all_data_blocks.append(np.array(current_block))
                        line_counts.append(current_line_count)
                        logger.debug(f"Saved block ending at line {line_number - 1} with {current_line_count} lines.")
                    current_block = []
                    current_line_count = 0
                    inside_block = True
                    logger.debug(f"Started new block at line {line_number}.")
                elif stripped_line == 'e':
                    # End of the current block
                    if inside_block and current_block:
                        all_data_blocks.append(np.array(current_block))
                        line_counts.append(current_line_count)
                        logger.debug(f"Ended block at line {line_number} with {current_line_count} lines.")
                    current_block = []
                    current_line_count = 0
                    inside_block = False
                elif inside_block:
                    # Collect data within the block
                    try:
                        numerical_values = list(map(float, stripped_line.split()))
                        current_block.append(numerical_values)
                        current_line_count += 1
                        logger.debug(f"Added line {line_number} to current block.")
                    except ValueError:
                        logger.warning(f"Non-numerical data encountered at line {line_number}: '{stripped_line}'. Skipping line.")
                        continue

            # After processing all lines, check if there's any remaining data in the current block
            if inside_block and current_block:
                all_data_blocks.append(np.array(current_block))
                line_counts.append(current_line_count)
                logger.debug(f"Saved final block ending at line {line_number} with {current_line_count} lines.")
    except FileNotFoundError as e:
        logger.error(f"File '{file_path}' not found.")
        raise e

    return all_data_blocks, line_counts

def compute_cdf(density_values: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
    """
    Compute the Cumulative Distribution Function (CDF) from a density array.

    This function takes an array of density values, calculates the cumulative sum,
    and normalizes it to produce the Cumulative Distribution Function (CDF).

    Parameters
    ----------
    density_values : list or numpy.ndarray
        An array of density values. Each element represents the density at a specific point.

    Returns
    -------
    cdf : list or numpy.ndarray
        An array of the same type as the input, where each element represents the
        cumulative sum of the density values up to that point, normalized to range [0, 1].

    Raises
    ------
    ValueError
        If the total sum of `density_values` is zero, making it impossible to compute a valid CDF.
    """
    total = np.sum(density_values)
    if total == 0:
        raise ValueError("Total density is zero, cannot compute CDF.")
    cdf = np.cumsum(density_values) / total
    return cdf

def inverse_transform_sampling(cdf: np.ndarray, grid_points: np.ndarray, num_samples: int ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform inverse transform sampling using the Cumulative Distribution Function (CDF).

    This function generates samples from a discrete distribution defined by the 
    provided CDF and corresponding grid points. It maps uniformly distributed 
    random samples to the grid points based on the inverse CDF method.

    Parameters
    ----------
    cdf : numpy.ndarray
        A one-dimensional array of cumulative distribution function values. 
        It must be monotonically increasing and normalized such that the last 
        element is 1.0.
    grid_points : numpy.ndarray
        A one-dimensional array of grid points corresponding to the CDF values. 
        The length of `grid_points` must match the length of `cdf`.
    num_samples : int
        The number of samples to generate.

    Returns
    -------
    sampled_grid_points : numpy.ndarray
        An array of sampled grid points based on the inverse transform sampling.
    sampled_indices : numpy.ndarray
        An array of indices corresponding to the sampled grid points.

    Raises
    ------
    ValueError
        If `cdf` is not one-dimensional.
    ValueError
        If `grid_points` is not one-dimensional.
    ValueError
        If the lengths of `cdf` and `grid_points` do not match.
    ValueError
        If `cdf` is not monotonically increasing.
    ValueError
        If the last element of `cdf` is not approximately 1.0.
    ValueError
        If `num_samples` is not a positive integer.

    Notes
    -----
    - The function assumes that the CDF is properly normalized.
    - It uses `numpy.searchsorted` to efficiently map uniform samples to grid indices.
    - The sampled grid points are selected based on the nearest grid index where 
      the CDF exceeds the random sample.

    """
    # Input Validation
    if not isinstance(cdf, np.ndarray):
        raise TypeError(f"Expected `cdf` to be a numpy.ndarray, got {type(cdf)} instead.")
    if not isinstance(grid_points, np.ndarray):
        raise TypeError(f"Expected `grid_points` to be a numpy.ndarray, got {type(grid_points)} instead.")
    if cdf.ndim != 1:
        raise ValueError(f"`cdf` must be one-dimensional, but has {cdf.ndim} dimensions.")
    if grid_points.ndim != 1:
        raise ValueError(f"`grid_points` must be one-dimensional, but has {grid_points.ndim} dimensions.")
    if len(cdf) != len(grid_points):
        raise ValueError(f"The length of `cdf` ({len(cdf)}) does not match the length of `grid_points` ({len(grid_points)}).")
    if not np.all(np.diff(cdf) >= 0):
        raise ValueError("`cdf` must be monotonically increasing.")
    if not np.isclose(cdf[-1], 1.0, atol=1e-8):
        raise ValueError(f"The last element of `cdf` must be approximately 1.0, but got {cdf[-1]}.")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError(f"`num_samples` must be a positive integer, but got {num_samples}.")

    # Generate uniform random samples in [0, 1)
    random_samples = np.random.rand(num_samples)

    # Map the random samples to grid indices using the inverse CDF
    sampled_indices = np.searchsorted(cdf, random_samples, side='right')

    # Ensure indices are within the valid range [0, len(grid_points) - 1]
    sampled_indices = np.clip(sampled_indices, 0, len(grid_points) - 1)

    # Retrieve the sampled grid points based on the indices
    sampled_grid_points = grid_points[sampled_indices]

    return sampled_grid_points, sampled_indices

def sample_n_dofs(
    full_1dden_data: List[Any],
    num_samples: int,
    idx_time: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform sampling for multiple degrees of freedom (DoFs) using densities and grid points.

    This function iterates over each degree of freedom, computes the cumulative distribution 
    function (CDF) based on provided density values, and performs inverse transform sampling 
    to select grid points and their corresponding indices. The sampled grid points and 
    indices for all DoFs are then combined into configurational space arrays.

    Parameters
    ----------
    full_1dden_data : List[Any]
        A list where each element corresponds to a degree of freedom (DoF). 
        Each DoF contains data structures (e.g., lists or arrays) indexed by `idx_time`, 
        where each time step includes a 2D array with grid points and their associated densities.
    num_samples : int
        The number of samples to draw for each degree of freedom.
    idx_time : int
        The index representing the specific time step to extract data from `full_1dden_data`.

    Returns
    -------
    sampled_grid_points : numpy.ndarray
        A 2D array where each row corresponds to sampled grid points across all DoFs.
    sampled_indices : numpy.ndarray
        A 2D array where each row corresponds to the indices of the sampled grid points 
        in the full configurational space.

    Raises
    ------
    ValueError
        If `full_1dden_data` is empty or `num_samples` is not positive.
    IndexError
        If `idx_time` is out of bounds for any DoF in `full_1dden_data`.
    """
    if not full_1dden_data:
        raise ValueError("`full_1dden_data` is empty. Provide valid density and grid data.")

    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("`num_samples` must be a positive integer.")

    n_dofs = len(full_1dden_data)  # Number of degrees of freedom (DoFs)
    sampled_grid_points_data: List[np.ndarray] = []
    sampled_indices_data: List[np.ndarray] = []

    logger.debug(f"Starting sampling for {n_dofs} degrees of freedom with {num_samples} samples each.")

    for dof in range(n_dofs):
        try:
            # Extract densities and grid points for this DoF at the specified time index
            dof_data = full_1dden_data[dof]
            time_step_data = dof_data[idx_time]
            densities = time_step_data[:, 1]
            grid_points = time_step_data[:, 0]

            logger.debug(f"DoF {dof + 1}: Extracted {len(densities)} density values and grid points.")
        except IndexError as e:
            logger.error(f"DoF {dof + 1}: Time index {idx_time} is out of bounds.")
            raise IndexError(f"DoF {dof + 1}: Time index {idx_time} is out of bounds.") from e
        except Exception as e:
            logger.error(f"DoF {dof + 1}: Unexpected error during data extraction: {e}")
            raise e

        try:
            # Compute the CDF for the current DoF
            cdf = compute_cdf(densities)
            logger.debug(f"DoF {dof + 1}: Computed CDF successfully.")
        except ValueError as e:
            logger.error(f"DoF {dof + 1}: Error computing CDF: {e}")
            continue  # Skip this DoF and proceed with others

        try:
            # Perform inverse transform sampling for the current DoF
            _, sampled_indices = inverse_transform_sampling(cdf, grid_points, num_samples)
            sampled_grid = grid_points[sampled_indices]

            logger.debug(f"DoF {dof + 1}: Performed inverse transform sampling successfully.")
        except Exception as e:
            logger.error(f"DoF {dof + 1}: Error during inverse transform sampling: {e}")
            continue  # Skip this DoF and proceed with others

        # Store the sampled grid points and indices for this DoF
        sampled_grid_points_data.append(sampled_grid)
        sampled_indices_data.append(sampled_indices)

    if not sampled_grid_points_data or not sampled_indices_data:
        logger.warning("No samples were collected. Check input data and sampling parameters.")
        return np.array([]), np.array([])

    try:
        # Combine the samples for all DoFs into a full configurational space
        sampled_grid_points = np.column_stack(sampled_grid_points_data)
        sampled_indices = np.column_stack(sampled_indices_data)
        logger.debug("Combined sampled data across all DoFs successfully.")
    except ValueError as e:
        logger.error(f"Error combining sampled data: {e}")
        raise ValueError(f"Error combining sampled data: {e}") from e

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

    Parameters
    ----------
    output_str : str
        The output string from the Fortran executable.

    Returns
    -------
    List[complex]
        A list of complex psi values for each state.

    Raises
    ------
    ValueError
        If no psi values are found in the output string.
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

def cluster_sampling(
    indices: np.ndarray,
    grid_points: np.ndarray,
    n_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply K-means clustering to the indices and return matching grid points and indices.

    Parameters
    ----------
    indices : numpy.ndarray
        A 2D numpy array containing the sampled indices used for clustering.
        Shape: (n_samples, n_features)
    grid_points : numpy.ndarray
        A 2D numpy array containing the corresponding grid points.
        Shape: (n_samples, n_features)
    n_clusters : int
        Number of clusters to create for representative sampling.

    Returns
    -------
    representative_grid_points : numpy.ndarray
        A numpy array of representative grid points from each cluster.
        Shape: (n_clusters, n_features)
    representative_indices : numpy.ndarray
        A numpy array of representative indices from each cluster (adjusted for Fortran indexing).
        Shape: (n_clusters, n_features)

    Notes
    -----
    - If the number of samples is less than the number of clusters, the number of clusters is reduced to the number of samples.
    - The indices are adjusted for Fortran indexing by adding 1 (since Fortran arrays are 1-based).
    """
    n_samples = indices.shape[0]
    if n_samples < n_clusters:
        logger.warning(
            f"Number of clusters ({n_clusters}) is greater than available samples ({n_samples}). "
            f"Reducing clusters to {n_samples}."
        )
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
        closest_sample_idx_in_cluster = np.argmin(distances)
        closest_sample_idx = cluster_sample_indices[closest_sample_idx_in_cluster]

        # Append the representative sample
        representative_indices.append(indices[closest_sample_idx])
        representative_grid_points.append(grid_points[closest_sample_idx])

    # Convert to numpy arrays
    representative_indices = np.array(representative_indices)
    representative_grid_points = np.array(representative_grid_points)

    # Adjust indices for Fortran indexing (starting from 1)
    representative_indices += 1

    return representative_grid_points, representative_indices


def save_samples_and_indices(
    grid_samples: np.ndarray, 
    index_samples: np.ndarray, 
    filename_prefix: str = 'mcmc'
) -> None:
    """
    Save both grid points and indices to separate NumPy binary files.

    This function saves two NumPy arrays, `grid_samples` and `index_samples`,
    to disk as `.npy` files. These arrays are saved with filenames that include
    a common prefix (provided by the user) followed by a descriptor for 
    the grid points and index samples.

    Parameters
    ----------
    grid_samples : np.ndarray
        A 2D numpy array containing the grid points sampled during the simulation.
        Each row represents a sample, and each column corresponds to a degree of freedom.
    index_samples : np.ndarray
        A 2D numpy array containing the corresponding indices of the sampled grid points.
        Each row represents a sample, and each column corresponds to the index in 
        the full configurational space.
    filename_prefix : str, optional
        The prefix for the saved filenames. Default is `'mcmc'`.

    Returns
    -------
    None

    Notes
    -----
    - This function saves two files: one for the grid samples and one for the index samples.
    - The files are saved in NumPy's binary `.npy` format.
    - The filenames follow the format:
        - `<filename_prefix>_grid_samples.npy`
        - `<filename_prefix>_index_samples.npy`

    Examples
    --------
    >>> grid_samples = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> index_samples = np.array([[1, 2], [3, 4], [5, 6]])
    >>> save_samples_and_indices(grid_samples, index_samples, filename_prefix='simulation_1')

    This will save two files:
    - `simulation_1_grid_samples.npy`
    - `simulation_1_index_samples.npy`
    """
    # Define filenames based on the provided prefix
    grid_filename = f'{filename_prefix}_grid_samples.npy'
    index_filename = f'{filename_prefix}_index_samples.npy'
    
    # Save grid samples and index samples as .npy files
    np.save(grid_filename, grid_samples)
    np.save(index_filename, index_samples)
    
    # Log the saved file locations
    logger.info(f"Grid samples saved to '{grid_filename}'")
    logger.info(f"Index samples saved to '{index_filename}'")

def run_and_collect(
    executable_path: str, 
    flags: List[str], 
    grid_indices: List[int],
    state_of_interest: int,
) -> Optional[float]:
    """
    Runs the Fortran executable with specified flags and grid indices, 
    then collects and calculates the sum of squared absolute psi values.

    This function passes grid indices and flags to a Fortran executable,
    retrieves the resulting psi values, and computes the sum of the squared 
    absolute values (representing the probability densities). It returns 
    the sum of these values or `None` if the execution fails.

    Parameters
    ----------
    executable_path : str
        Path to the Fortran executable that computes the psi values.
    flags : List[str]
        A list of command-line flags to pass to the Fortran executable.
    grid_indices : List[int]
        A list of grid indices (integers) that define the grid point 
        to evaluate in the Fortran executable.
    state_of_interest : int
        The state for which the density is to be computed.

    Returns
    -------
    Optional[float]
        The sum of squared absolute psi values (probability densities), 
        or `None` if the execution of the Fortran executable fails.

    Notes
    -----
    - The `run_fortran_executable` function must return an array of psi values.
    - This function assumes that the psi values returned from the Fortran 
      executable are complex numbers, and it calculates the sum of their squared magnitudes.
    - The returned value represents the sum of probabilities across the states.

    """
    # Run the Fortran executable and collect psi values
    psi_values = run_fortran_executable(executable_path, flags, grid_indices)
    if psi_values is not None:
        if state_of_interest == -1:
            # Compute the sum of squared absolute psi values for all states
            psi_squared = sum(abs(psi)**2 for psi in psi_values)
        else:
            # Compute the sum of squared absolute psi values for the specified state
            psi_squared = abs(psi_values[state_of_interest])**2
        return psi_squared
            
    else:
        # Return None if the executable failed
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


def get_neighbors(
    index: Tuple[int, ...],
    shape: Tuple[int, ...],
    boundary: str = 'reflect',
    max_step: int = 1
) -> List[Tuple[int, ...]]:
    """
    Get neighboring indices on a multidimensional grid with specified boundary conditions.

    This function calculates the valid neighboring indices around a given index 
    in a grid, considering the specified boundary conditions ('reflect' or 'periodic'). 
    It returns all neighboring points within the specified `max_step` distance in 
    each dimension.

    Parameters
    ----------
    index : Tuple[int, ...]
        A tuple representing the current index in the grid. The length of this tuple 
        should match the number of dimensions of the grid (i.e., `len(index) == len(shape)`).
    shape : Tuple[int, ...]
        A tuple representing the shape of the grid, where each element is the size of 
        the grid in that dimension.
    boundary : str, optional
        Specifies the type of boundary conditions. Can be either 'reflect' or 'periodic'.
        Default is 'reflect'. 
        - 'reflect': Reflects off the boundaries (i.e., no values outside the grid).
        - 'periodic': Wraps around the grid (i.e., opposite edges are connected).
    max_step : int, optional
        Maximum step size in any dimension to consider as a neighbor. Default is 1, 
        which means all neighbors with a step size of ±1 in each dimension will be returned.

    Returns
    -------
    List[Tuple[int, ...]]
        A list of tuples, where each tuple represents the valid neighboring indices 
        around the given `index`. The list includes neighbors from all dimensions, 
        excluding the original position.

    Raises
    ------
    ValueError
        If an unknown boundary condition is specified.

    Notes
    -----
    - This function works for grids of arbitrary dimensionality.
    - The center index (i.e., the input index itself) is not included in the returned neighbors.
    """
    neighbors = []
    index = np.array(index)
    dims = len(index)
    deltas = range(-max_step, max_step + 1)

    # Generate all possible combinations of steps in each dimension
    delta_grids = np.meshgrid(*([deltas] * dims))  # Create grid of deltas
    delta_combinations = np.array(delta_grids).T.reshape(-1, dims)  # Flatten combinations

    for delta in delta_combinations:
        if np.all(delta == 0):
            continue  # Skip the original position (the center)

        neighbor = index + delta  # Calculate the neighbor index
        valid = True

        for d in range(dims):
            if neighbor[d] < 0 or neighbor[d] >= shape[d]:  # Check if out of bounds
                if boundary == 'periodic':
                    neighbor[d] %= shape[d]  # Apply periodic boundary
                elif boundary == 'reflect':
                    neighbor[d] = max(0, min(neighbor[d], shape[d] - 1))  # Apply reflective boundary
                else:
                    logger.error(f"Unknown boundary condition: '{boundary}'")
                    raise ValueError(f"Unknown boundary condition: '{boundary}'")

        if valid:
            neighbors.append(tuple(neighbor))  # Add valid neighbors as a tuple

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


def metropolis_worker(
    args: Tuple[
        np.ndarray,        # initial_index
        float,             # initial_prob
        Tuple[int, ...],   # grid_shape
        Callable,          # run_and_collect_fn
        str,               # executable_path
        List[str],         # flags
        int,               # state_of_interest
        int,               # num_steps
        str,               # boundary
        dict               # shared_cache
    ]
) -> List[Tuple[np.ndarray, float]]:
    """
    Worker function for performing Metropolis sampling in a multiprocessing environment.

    This function performs a Metropolis sampling algorithm on a multidimensional grid. 
    It proposes new positions by selecting neighboring grid points, computes their 
    densities using a provided function, and decides whether to accept or reject the 
    proposed position based on the Metropolis acceptance criterion.

    Parameters
    ----------
    args : tuple
        A tuple containing the following elements:
        - initial_index (np.ndarray): The initial grid index for the worker.
        - initial_prob (float): The initial probability (density) at the starting position.
        - grid_shape (tuple): The shape of the grid (number of points in each dimension).
        - run_and_collect_fn (Callable): Function to compute the density for a given grid index.
        - state_of_interest (int): In a multi state system, the state for which the density is to be computed. (-1 for all states)
        - executable_path (str): Path to the executable used for computing densities.
        - flags (list of str): Command-line flags to pass to the executable.
        - num_steps (int): Number of steps the Metropolis algorithm will take.
        - boundary (str): Type of boundary condition ('reflect' or 'periodic').
        - shared_cache (dict): Shared cache to store already computed densities for grid points.

    Returns
    -------
    List[Tuple[np.ndarray, float]]
        A list of tuples, where each tuple contains:
        - The accepted grid index (np.ndarray).
        - The corresponding probability (density) value (float).

    Notes
    -----
    - The `run_and_collect_fn` function is expected to compute and return the probability density 
      for a given grid index.
    - The worker maintains a cache of computed densities to avoid redundant calculations.
    - If a proposed move leads to a position with zero probability or invalid density, it is skipped.
    - The acceptance criterion for a move is based on the ratio of proposed to current density.

    """
    (initial_index, initial_prob, grid_shape, run_and_collect_fn, executable_path, flags, state_of_interest, num_steps, boundary, shared_cache) = args
    current_index = initial_index
    current_prob = initial_prob
    samples = [(current_index, current_prob)]  # Store tuples of (position, density)
    cache = shared_cache

    for _ in range(num_steps):
        # Propose a neighbor
        neighbors = get_neighbors(current_index, grid_shape, boundary, max_step=2)
        if not neighbors:
            continue  # No valid neighbors, skip this iteration

        # Randomly select a neighbor as the proposed move
        proposed_index = neighbors[np.random.randint(len(neighbors))]

        # Check if the proposed position is already in the cache
        proposed_key = tuple(proposed_index)
        new_den = cache.get(proposed_key)

        if new_den is None:
            # Compute the new density using the provided function
            new_den = run_and_collect_fn(executable_path, flags, list(proposed_index),state_of_interest)
            if new_den is None:
                continue  # Skip if density could not be computed
            cache[proposed_key] = new_den  # Store the new density in the cache

        # Skip if densities are zero (can't compute acceptance ratio)
        if new_den == 0 or current_prob == 0:
            continue

        # Compute the acceptance probability
        acceptance = new_den / current_prob

        # Accept the new position with a probability proportional to the acceptance ratio
        if acceptance >= 1 or np.random.rand() < acceptance:
            current_index = proposed_index
            current_prob = new_den
            samples.append((current_index, current_prob))  # Store the accepted position and density

    return samples

def metropolis_parallel(
    initial_indices: List[np.ndarray],
    initial_sp_psi: List[float],
    grid_shape: Tuple[int, ...],
    run_and_collect_fn: Callable,
    executable_path: str,
    flags: List[str],
    state_of_interest: int,
    num_steps: int,
    boundary: str = 'reflect'
) -> List[List[Tuple[np.ndarray, float]]]:
    """
    Parallel Metropolis algorithm using multiprocessing.

    This function runs multiple instances of the Metropolis algorithm in parallel, 
    each starting from a different initial position (walker). It uses multiprocessing 
    to distribute the workload across multiple CPU cores, with a shared cache to 
    store already computed densities to avoid redundant calculations.

    Parameters
    ----------
    initial_indices : List[np.ndarray]
        A list of initial grid indices for each walker. Each element is an array 
        representing the starting position for a specific walker.
    initial_sp_psi : List[float]
        A list of initial probability densities (psi values) corresponding to each 
        walker's starting position.
    grid_shape : Tuple[int, ...]
        The shape of the grid (number of points in each dimension).
    run_and_collect_fn : Callable
        A function that computes the probability density for a given grid index.
    executable_path : str
        Path to the Fortran executable that computes psi values.
    flags : List[str]
        Command-line flags to pass to the Fortran executable.
    state_of_interest : int
        The state for which the density is to be computed. (-1 for all states)
    num_steps : int
        The number of steps that each walker will take in the Metropolis algorithm.
    boundary : str, optional
        The type of boundary condition to use ('reflect' or 'periodic'). Default is 'reflect'.

    Returns
    -------
    List[List[Tuple[np.ndarray, float]]]
        A list where each element is a list of samples from one worker. Each sample is 
        represented as a tuple containing the grid index (np.ndarray) and the 
        corresponding probability density (float).

    Notes
    -----
    - This function uses a shared cache (dictionary) to store computed densities across 
      all workers, avoiding redundant calculations.
    - The function relies on the `metropolis_worker` function to perform the Metropolis 
      sampling for each walker.
    - The number of parallel processes is determined by the number of CPU cores available.
    """
    N = len(initial_indices)  # Number of walkers

    # Shared cache for densities across all workers
    manager = mp.Manager()
    shared_cache = manager.dict()

    # Prepare arguments for each worker (one for each walker)
    args_list = [
        (
            initial_indices[i],   # Initial grid index for this walker
            initial_sp_psi[i],    # Initial probability density for this walker
            grid_shape,           # Shape of the grid
            run_and_collect_fn,   # Function to compute density
            executable_path,      # Path to the Fortran executable
            flags,                # Command-line flags
            state_of_interest,    # Command-line flags
            num_steps,            # Number of Metropolis steps
            boundary,             # Boundary condition ('reflect' or 'periodic')
            shared_cache          # Shared cache for storing computed densities
        )
        for i in range(N)
    ]

    # Use a multiprocessing pool to run the Metropolis algorithm in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(metropolis_worker, args_list)

    return results