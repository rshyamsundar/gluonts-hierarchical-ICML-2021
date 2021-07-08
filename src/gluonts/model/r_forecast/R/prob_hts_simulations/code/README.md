# Source code for the simulation experiments of the paper *Hierarchical Probabilistic Forecasting of Electricity Demand with Smart Meter Data* by Ben Taieb, Souhaib, Taylor, James, and Hyndman, Rob.


The code implements all the results for the simulation experiments in Section 7.

# Usage

1. Specify the scenario in *runsim.sh*, i.e. specify experiment, marg, Tlearning, M, Trueparam, etc. See runsim.sh for some examples.
2. Run the script *runsim.sh*
```
./runsim.sh
```
3. To produce the figures with the results, run the script *analysis.R* with the right scenario, as in step 1.
```
./analysis.sh
```
4. The results will then be available in the folder *work*.
