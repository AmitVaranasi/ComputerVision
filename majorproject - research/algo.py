def sequence_alignment(x, y, delta, alpha):
    """
    Computes the optimal alignment of two sequences x and y using dynamic programming.

    Parameters:
    - x: List or string representing the first sequence.
    - y: List or string representing the second sequence.
    - delta: Gap penalty (a non-negative number).
    - alpha: A function that takes two characters and returns the mismatch penalty.

    Returns:
    - A tuple containing:
        - The optimal alignment cost (an integer or float).
        - The aligned sequence for x (a string).
        - The aligned sequence for y (a string).
    """
    m = len(x)
    n = len(y)
    
    # Initialize M (cost matrix) and P (pointer matrix)
    M = [[0] * (n + 1) for _ in range(m + 1)]
    P = [[''] * (n + 1) for _ in range(m + 1)]
    
    # Base cases: filling the first row and first column
    for i in range(m + 1):
        M[i][0] = i * delta
        P[i][0] = 'up' if i > 0 else 'none'
    
    for j in range(n + 1):
        M[0][j] = j * delta
        P[0][j] = 'left' if j > 0 else 'none'
    
    # Fill M and P matrices using dynamic programming
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Compute costs for different moves
            match_cost = alpha(x[i - 1], y[j - 1]) + M[i - 1][j - 1]
            delete_cost = delta + M[i - 1][j]
            insert_cost = delta + M[i][j - 1]
            
            # Choose the minimum cost
            M[i][j] = min(match_cost, delete_cost, insert_cost)
            
            # Record the move in P matrix
            if M[i][j] == match_cost:
                P[i][j] = 'diag'
            elif M[i][j] == delete_cost:
                P[i][j] = 'up'
            else:
                P[i][j] = 'left'
    
    # Reconstruct the alignment from P matrix
    aligned_x = []
    aligned_y = []
    i, j = m, n
    while i > 0 or j > 0:
        direction = P[i][j]
        if direction == 'diag':
            aligned_x.append(x[i - 1])
            aligned_y.append(y[j - 1])
            i -= 1
            j -= 1
        elif direction == 'up':
            aligned_x.append(x[i - 1])
            aligned_y.append('-')
            i -= 1
        elif direction == 'left':
            aligned_x.append('-')
            aligned_y.append(y[j - 1])
            j -= 1
        else:
            break  # Reached the starting cell
    
    # The sequences are built backwards, so reverse them
    aligned_x.reverse()
    aligned_y.reverse()
    
    # Convert lists to strings
    aligned_x_str = ''.join(aligned_x)
    aligned_y_str = ''.join(aligned_y)
    
    return M[m][n], aligned_x_str, aligned_y_str

# Example usage:
def mismatch_penalty(a, b):
    return 0 if a == b else 1  # Simple mismatch penalty function

# Sample sequences
x_sequence = "AGACTAGTTAC"
y_sequence = "CGAGACGT"

# Gap penalty
gap_penalty = 2

# Compute the alignment
cost, aligned_x, aligned_y = sequence_alignment(x_sequence, y_sequence, gap_penalty, mismatch_penalty)

# Display the results
print(f"Optimal alignment cost: {cost}")
print(f"Aligned Sequence X: {aligned_x}")
print(f"Aligned Sequence Y: {aligned_y}")