"""Mirage: does the execution reward you train on tell the truth?

A reward-fidelity study for GRPO text-to-SQL. The package is split so that
everything that decides whether a number is *right* (data, execution, result
comparison, reward, evaluation, statistics) runs on a CPU with no ML libraries,
and everything that needs a GPU (training, generation) is imported lazily.
"""

__version__ = "0.1.0"
