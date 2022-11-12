# MyCode

## DATAFRAME

### Compter le nombre de lignes d'un dataframe

```
rows = len(df)
rows = len(df.axes[0])
rows = df.shape[0]
```
### Compter le nombre d'occurence des valeurs d'une colonne

```
df['col'].value_counts()
```
### Reset index of a dataframe 

```
df = df.reset_index()
```

### Rename columns

```
df.columns = ['col1', 'col2']
```
### make random function be repeatable

> Set random seed to x=`334` (x value doesn't matter)

```
np.random.seed(334)
```

### Obtenir un échantillon d'un dataframe de n lignes avec ou sans remplacement

> replace = `True` si on remet en jeu l'élément tiré, faux sinon

```
sample = df.sample(n=5, replace=False)
sample = df.sample(n=10, replace=True)
```

### Return evenly spaced numbers over a specified interval.

```
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
```

### Compute the arithmetic mean along the specified axis.

```
numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)
```
### Compute the standard deviation along the specified axis.

```
numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>)
```

## MATPLOTLIB

### Clear the current figure.

> matplotlib.pyplot.clf()

```
import matplotlib.pyplot as plt
plt.clf()
```

### Histogram of a dataframe column

```
df['column'].hist() # or df.hist('column') : to check
plt.show()
```

## DISCRETE DISTRIBTION

### Expected value (discrete distribution)

> Remember that expected value can be calculated by multiplying each possible outcome with its corresponding probability and taking the sum

```
expected_value = np.sum(df['column'] * df['prob'])
```

### Create probability distribution (discrete distribution)

```
counts_by_value = df['column'].value_counts()
print(counts_by_value)
row_count = len(df)
print(row_count)
dist = counts_by_value / row_count
print(dist)
```

## CONTINUOUS UNIFORM DISTRIBTION

### Continuous uniform distribution

> * probability is the area
> * cdf = cumulative distribution function : `P(variable<=x)`
> * uniform.cdf(`x`, loc=`0`, scale=`1`)
> * uniform.cdf(`less or equal than x`, `minimum value`, `maximum value`)

```
from scipy.stats import uniform
uniform.cdf(7, 0, 12)
```

### Generating random numbers according to uniform distribution

> * uniform.rvs(loc=`0`, scale=`1`, size=`1`, random_state=`None`)
> * uniform.rvs(`minimum value`, `maximum value`, size=`number of values to generate`)

```
from scipy.stats import uniform
uniform.rvs(0, 5, size=10)
```

### Exemple de generation d'un distribution continue uniforme

```
uniform_continuous_distibution=True
if uniform_continuous_distibution==True:

    # import numpy
    import numpy as np
    
    # Set random seed to 334
    np.random.seed(334)

    # Import uniform
    from scipy.stats import uniform

    # Generate 1000 wait times between 0 and 30 mins
    wait_times = uniform.rvs(0, 30, size=1000)

    # import matplotlib
    import matplotlib.pyplot as plt

    # Create a histogram of simulated times and show plot
    plt.clf()
    plt.hist(wait_times)
    plt.show()
```

## BINOMIAL DISTRIBTION

> * Binomial distribution = probability distribution of the number of successes in a sequence of independant trials
> * Example : number of heads in a sequence of coin flips
> * n = total number of trials being performed
> * p = probability of success

### Generating random numbers according to binomial distribution

> * binom.rvs(`number of coins`, `probability of heads/success`, size=`number of trials`)

```
from scipy.stats import binom
binom.rvs(1, 0.5, size=10)
binom.rvs(10, 0.5, size=1)
binom.rvs(10, 0.25, size=10)
```

### probability of x heads : P(heads=x)

> * binom.pmf(k, n, p, loc=`0`)
> * binom.pmf(x=`num heads`, n=`num trials`, p=`prob of heads`)
> * pmf = Probability mass function

```
binom.pmf(7, 10, 0.5)
```

### probability of less or equal than x heads : P(heads<=x)

> * binom.cdf(k, n, p, loc=0)
> * binom.cdf(<x=num heads>, <n=num trials>, <p=prob of heads>)
> * cdf = cumulative distribution function

```
binom.cdf(7, 10, 0.5)
```

### Expected value of binomial distribution : n x p

Example :

```
binomial_distibution=True
if binomial_distibution==True:

    # Expected number won with 30% win rate
    won_30pct = 3 * 0.3
    print(won_30pct)

    # Expected number won with 25% win rate
    won_25pct = 3 * 0.25
    print(won_25pct)

    # Expected number won with 35% win rate
    won_35pct = 3 * 0.35
    print(won_35pct)
```

## NORMAL DISTRIBTION

### Distribution symetrique

> * Aire en dessous de la courbe = 1
>   1. écart-type correspond à 68% de l'aire en dessous de la courbe
>   2. écart-type correspond à 95% de l'aire en dessous de la courbe
>   3. écart-type correspond à 99,7% de l'aire en dessous de la courbe
> * La courbe n'arrive jamais à 0 (la probabilité n'est jamais nulle)
> * 2 paramètres descriptifs : moyenne et écart-type
> * Standard normal distribution : Mean=0 and Standard deviation=1

### import scipy.stats.norm

```
from scipy.stats import norm
```

### probability of less or equal than x  : P(variable<=x)

> * norm.cdf(x, loc=`0`, scale=`1`)
> * norm.cdf(x, `mean`, `standard deviation`)
> * cdf = cumulative distribution function

```
norm.cdf(154, 161, 7)
```

### what height are 90% of women shorter than 
### wich value of variable corresponding to a given population percentage

> * ppf = Percent point function
> * norm.ppf(q, loc=`0`, scale=`1`)
> * norm.ppf(q=`% of population`, `mean`, `standard deviation`)

```
norm.ppf(0.9, 161, 7)
```

### what height are 90% of women taller than?

```
norm.ppf((1-0.9), 161, 7)
```

### Generating random numbers according to normal distribution

> * norm.rvs(loc=`0`, scale=`1`, size=`1`, random_state=`None`)
> * norm.rvs(`mean`, `standard deviation`, size=`number of values to generate`)

```
norm.rvs(161, 7, size=10)
```

## CENTRAL LIMIT DISTRIBUTION (CLT)

> * The sampling distribution of a statistic becomes closer to the normal distribution as the number of trials increases.
> * Samples should be random and independant (with replacement)

### Rolling a dice 5 times 1000 times

> * On jette 5 fois 1 dés et on calcule la moyenne des résultats
> * On ré-itère cette opération 1000 fois et on calcule la moyenne des moyennes

```
import numpy as np
sample_means = []
for i in range(1000):
    sample_5.append(np.mean(samp_5))
print(sample_means)
```

### Idem mais avec l'écart-type au lieu de la moyenne

```
import numpy as np
sample_sds = []
for i in range(1000):
    sample_sds.append(np.std(die.sample(5, replace-True)))
print(sample_sds)
```

### Idem pour les proportions (to check / not sure)

```
import numpy as np
sales_teams = pd.Series(["Amir", "Brian", "Claire", "Damian"])
sample_teams_means = []
for i in range(1000):
    sample_teams_means.append(0.25 * sales_teams.sample(10, replace=True))
print(sample_teams_means)
```

### Huge population and don't have the time to collect data on every one

> * Collect small samples et calculate the mean or standard deviation distribution to approximate their values quite correctly

EXAMPLE : 

```
# Set seed to 104
np.random.seed(104)

sample_means = []
# Loop 100 times
for i in range(100):
  # Take sample of 20 num_users
  samp_20 = amir_deals['num_users'].sample(20, replace=True)
  # Calculate mean of samp_20
  samp_20_mean = np.mean(samp_20)
  # Append samp_20_mean to sample_means
  sample_means.append(samp_20_mean)
  
# Convert to Series and plot histogram
sample_means_series = pd.Series(sample_means)
sample_means_series.hist()
# Show plot
plt.show()
```

## POISSON DISTRIBUTION

### Poisson process :

> * Events appear to happen at a certain rate, but completely at random
> * Poisson distribution of some number of events occuring over a fixed period of time
> * Examples :
>   * Probability of >= 5 annimals adoped from an animal shelter per week
>   * Probability of 12 people arrinving at a restaurant per hour
>   * Probability of < 20 earthquakes in California per year
>  * A Poisson distribution is reprensented by Lambda = average number of events per time interval (expected value of the distribution)
> * Average number of adoptions per week = 8 
> * A Poisson distribution is a discrete distribution
> * The distribution peak is always at the lambda value

### P(Event = x) on a Poisson distribution with lambda=l

> * poisson.pmf(k, mu, loc=`0`)
> * poisson.pmf(x, `lambda`, `???`)
> * pmf = probability mass function
> * P(Number of adoptions in a week = 5) on a Poisson distribution with lambda=8

```
from scipy.stats import poisson
poisson.pmf(5, 8)
```

### P(Event <= x) on a Poisson distribution with lambda=l

> * poisson.cdf(k, mu, loc=`0`)
> * poisson.cdf(x, `lambda`, `???`)
> * cdf = cumulative distribution function
> * P(Number of adoptions in a week <= 5) on a Poisson distribution with lambda=`8`

```
from scipy.stats import poisson
poisson.cdf(5, 8)
```

### Generating random numbers according to Poisson distribution

> * poisson.rvs(`lambda`, size=`number of samples`)

```
from scipy.stats import poisson
poisson.rvs(8, size=10)
```

## EXPONENTIAL DISTRIBUTION

> * Probability of time between two Poisson events
> * Examples :
>   * Probability of 1 day between adoptions
>   * Probability of <10 min between restaurant arrivals
>   * Probability of 6-8 months between earthquakes
> * Also use lambda (rate : # events per unit of time)
>   * Example : lamba = 0.5 customer service tickets created each minute
> * Continuous distribution (time)

### Expected value of exponential distribution : 1/lambda (Example : 1 request per 2 minutes)

> * How long until a new event ? P(wait < x time unit)
> * expon.cdf(x, scale=`lambda`)

```
from scipy.stats import expon
expon.cdf(1, scale=0.5)
```

## STUDENT'S T-DISTRIBUTION

> * Has parameter degrees of freedom (df) which affects the thickness of the tails
> * Lower df : thicker tails, higher standard deviation
> * Higher df : closer to normal distribution

## LOG-NORMAL DISTRIBUTION

> * Variable whose logarithm is normally distributed
> * Examples : 
>   * Length of chess games
>   * Adult blood pressure
>   * Number of hospitalizations in the 2003 SARS outbreak

## CORRELATION

### Draw a scatter plot

> * sns.scatterplot(x=`"df column name for x axe"`, y=`"df column name for y axe"`, data=`dataframe`)

```
import seaborn as sns
sns.scatterplot(x="sleep_total", y="sleep_rem", data=msleep)
plt.show()
```

### Draw a scatter plot with regression line

```
import seaborn as sns
sns.lmplot(x="sleep_total", y="sleep_rem", data=msleep, ci=None)
plt.show()
```

### Computing correlation coefficient

> * df["`x column`"].corr(df["`y column`"])

```
msleep["sleep_total"].corr(msleep["sleep_rem"])
```

### Many ways to calculate correlation

> * Used in this course : Pearson product-moment correlation (rho)
>   * with x bar = `mean of x` and sigma x = `standard deviation of x`
>   * rho = sum( (xi-mean(x)) x (yi-mean(y)) / ( std(x) x std(y) ) )

> * Variation on this formula :
>   * Kendall's tau
>   * Spearman Rho

## CORRELATION CAVEAT

### Non linear correlation

* Some correlations are not linear
* Example : log, quatratic, exp...
* A low correlation coefficient doesn't mean that there is no correlation but no linear correletion

### Variable transformation to return into linear regression correlation case

np.log(x), sqrt(x), 1/x, ...

### Correlation doesn't mean causality

Margarine consumption increase is strongly correlated with divorce rate (r=0,99) but it doesn't mean eating more margarine implies a divorce !

# Confounders

Une correlation directe entre 2 variables peut masquer une 3ème variable qui est la variable avec un vrai lien de causalité
Exemple : Les cancers du poumons sont correlés avec la quantité de café bue. En fait, ceux qui boivent du café fume aussi des cigarettes... 
et ce sont ces cigarettes qui ont un vrai impact (relation de causalité) avec les cancers du poumon...
Les cigarettes sont appelés "cofounders" (co-facteurs j'imagine)



