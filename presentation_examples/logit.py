import numpy as np
import matplotlib.pyplot as plt

# Logit range
logits = np.linspace(-10, 10, 500)

# Sigmoid function
prob = 1 / (1 + np.exp(-logits))

# Derivative of sigmoid (slope)
slope = prob * (1 - prob)

# Plot
plt.figure()
plt.plot(logits, prob, label="Probability (sigmoid)")
plt.plot(logits, slope, label="Slope dP/dLogit", linestyle="--")
plt.xlabel("Logit")
plt.ylabel("Value")
plt.title("Logit → Probability and Sensitivity")
plt.legend()
plt.grid(True)
plt.show()
