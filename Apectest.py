from aphec import aphex_equation
import numpy as np
import sys

def run_test(alpha_val=0.1, delay_scale=0.001):
    """Run test with configurable parameters"""
    D = np.array([[delay_scale, delay_scale*2, delay_scale*3]])
    F = np.array([
        [0.2, 0.1, 0.05],
        [0.05, 0.2, 0.1],
        [0.1, 0.05, 0.2]
    ])
    F_ext = np.array([0.001, 0.001, 0.001])
    result = aphex_equation(D, F, F_ext, alpha_val)
    print(f"Test (alpha={alpha_val}, delay={delay_scale}):", result)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Get alpha from command line
        alpha = float(sys.argv[1])
        # Get delay scale if provided
        delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
        run_test(alpha, delay)
    else:
        # Run a batch of tests
        print("Running test batch...")
        alphas = [0.1, 0.3, 0.5]
        delays = [0.001, 0.01, 0.1]
        for a in alphas:
            for d in delays:
                run_test(a, d)