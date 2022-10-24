## =============================================================
## NOTES PYTHON
## =============================================================

# Compter le nombre de lignes d'un dataframe
rows = len(df)
rows = len(df.axes[0])
rows = df.shape[0]

# Compter le nombre d'occurence des valeurs d'une colonne
df['col'].value_counts()

# Reset index of a dataframe 
df = df.reset_index()

# Rename columns
df.columns = ['col1', 'col2']

# make random function be repeatable
# Set random seed to x=334
# x value doesn't matter
np.random.seed(334)

# Obtenir un échantillon d'un dataframe de n lignes 
# avec ou sans remplacement
# replace = True si on remet en jeu l'élément tiré, faux sinon
sample = df.sample(n=5, replace=False)
sample = df.sample(n=10, replace=True)


# Return evenly spaced numbers over a specified interval.
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)

# Compute the arithmetic mean along the specified axis.
numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)

# Clear the current figure.
# matplotlib.pyplot.clf()
import matplotlib.pyplot as plt
plt.clf()

# Expected value (discrete distribution)
# Remember that expected value can be calculated by multiplying each possible outcome with its corresponding probability and taking the sum
expected_value = np.sum(df['column'] * df['prob'])

# Create probability distribution (discrete distribution)
counts_by_value = df['column'].value_counts()
print(counts_by_value)
row_count = len(df)
print(row_count)
dist = counts_by_value / row_count
print(dist)

# Continuous uniform distribution
# probability is the area
# cdf = cumulative distribution function
# uniform.cdf(x, loc=0, scale=1)
# uniform.cdf(<less or equal than x>, <minimum value>, <maximum value>)
from scipy.stats import uniform
uniform.cdf(7, 0, 12)

# Generating random numbers according to uniform distribution
# uniform.rvs(loc=0, scale=1, size=1, random_state=None)
# uniform.rvs(<minimum value>, <maximum value>, size=<number of values to generate>)
from scipy.stats import uniform
uniform.rvs(0, 5, size=10)

# Exemple de generation d'un distribution continue uniforme

    # Set random seed to 334
    np.random.seed(334)

    # Import uniform
    from scipy.stats import uniform

    # Generate 1000 wait times between 0 and 30 mins
    wait_times = uniform.rvs(0, 30, size=1000)

    # Create a histogram of simulated times and show plot
    plt.clf()
    plt.hist(wait_times)
    plt.show()

# Binomial distribution = probability distribution of the number of successes in a sequence of independant trials
#                         Example : number of heads in a sequence of coin flips
# n = total number of trials being performed
# p = probability of success

# Generating random numbers according to binomial distribution
# binom.rvs(<number of coins>, <probability of heads/success>, size=<number of trials>)
from scipy.stats import binom
binom.rvs(1, 0.5, size=10)
binom.rvs(10, 0.5, size=1)
binom.rvs(10, 0.25, size=10)

# probability of x heads : P(heads=x)
# binom.pmf(k, n, p, loc=0)
# binom.pmf(<x=num heads>, <n=num trials>, <p=prob of heads>)
# pmf = Probability mass function
binom.pmf(7, 10, 0.5)

# probability of less or equal than x heads : P(heads<=x)
# binom.cdf(k, n, p, loc=0)
# binom.cdf(<x=num heads>, <n=num trials>, <p=prob of heads>)
# cdf = cumulative distribution function
binom.cdf(7, 10, 0.5)

# Expected value of binomial distribution : n x p
# Example :
    # Expected number won with 30% win rate
    won_30pct = 3 * 0.3
    print(won_30pct)

    # Expected number won with 25% win rate
    won_25pct = 3 * 0.25
    print(won_25pct)

    # Expected number won with 35% win rate
    won_35pct = 3 * 0.35
    print(won_35pct)


