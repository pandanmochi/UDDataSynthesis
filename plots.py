import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest

# Generate sample data from a normal distribution
sample_data = np.random.normal(loc=0, scale=1, size=100)

# Calculate the empirical distribution function (ECDF)
def ecdf(data):
    x = np.sort(data)
    n = len(data)
    y = np.arange(1, n+1) / n
    return x, y

x_sample, y_sample = ecdf(sample_data)

# Calculate the theoretical cumulative distribution function (CDF)
x_theoretical = np.linspace(np.min(sample_data), np.max(sample_data), 1000)
y_theoretical = norm.cdf(x_theoretical, loc=0, scale=1)

# Plot the ECDF and CDF
plt.rc('font', family='serif')
plt.step(x_sample, y_sample, label='ECDF')
plt.plot(x_theoretical, y_theoretical, label='Theoretical CDF')
plt.legend()
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.title('Kolmogorov-Smirnov Test Visualization')

# Save the plot as a PDF
plt.savefig('ks_test_plot.pdf')
