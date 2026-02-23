import numpy as np
import matplotlib.pyplot as plt

# Probability range (avoid 0 and 1 to prevent infinities)
p = np.linspace(0.001, 0.999, 500)

# Convert probability to logit
logit = np.log(p / (1 - p))

# Apply +0.1 logit shift
logit_shifted = logit + 0.1

# Convert back to probability
p_shifted = 1 / (1 + np.exp(-logit_shifted))

# Probability change
delta_p = p_shifted - p

# Plot
plt.figure()
plt.plot(p, delta_p)
plt.xlabel("Original Probability")
plt.ylabel("Probability Increase (ΔP for +0.1 logit)")
plt.title("Effect of +0.1 Logit Increase Across Probabilities")
plt.grid(True)
plt.show()
