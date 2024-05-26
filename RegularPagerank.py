import numpy as np

def create_transition_matrix(links, n):
    """
    Create the transition matrix from the links.
    :param links: List of tuples where each tuple (i, j) represents a link from page i to page j.
    :param n: Number of pages
    :return: Transition matrix
    """
    M = np.zeros((n, n))
    for i, j in links:
        M[j, i] += 1

    # Normalize columns to sum to 1
    column_sums = M.sum(axis=0)
    for i in range(n):
        if column_sums[i] != 0:
            M[:, i] /= column_sums[i]
        else:
            M[:, i] = 1.0 / n  # Handle dangling nodes by distributing uniformly

    return M

def pagerank(links, n, d=0.85, max_iterations=100, tol=1.0e-6):
    """
    Compute the PageRank vector.
    :param links: List of tuples where each tuple (i, j) represents a link from page i to page j.
    :param n: Number of pages
    :param d: Damping factor (default 0.85)
    :param max_iterations: Maximum number of iterations (default 100)
    :param tol: Tolerance for convergence (default 1.0e-6)
    :return: PageRank vector
    """
    M = create_transition_matrix(links, n)
    rank = np.ones(n) / n
    for _ in range(max_iterations):
        new_rank = (1 - d) / n + d * M @ rank
        if np.linalg.norm(new_rank - rank, 1) < tol:
            return new_rank
        rank = new_rank
    return rank

# Example usage
if __name__ == "__main__":
    # List of links (from_page, to_page)
    links = [(0, 1), (0, 2), (1, 2), (2, 0), (2, 1)]
    n = 3  # Number of pages
    ranks = pagerank(links, n)
    print("PageRank scores:", ranks)
