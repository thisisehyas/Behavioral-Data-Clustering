import numpy as np

for alpha in ["0.1", "0.5", "1.0", "2.0", "5.0"]:
    matrix = np.load(
        f"analysis/alpha_sensitivity/aggregated/ari_matrix_alpha_{alpha}.npy"
    )

    print(f"Alpha: {alpha}")
    print(np.array_str(matrix, precision=4))
    print()