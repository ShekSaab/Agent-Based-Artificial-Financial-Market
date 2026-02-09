# Agent-Based-Artificial-Financial-Market
An artificial financial market simulation exploring the interactions between Zero-Intelligence Traders (ZIT) and Reinforcement Learning (RL) agents.

## Project Overview
This project simulates a continuous double-auction artificial financial market to study the interaction between **Zero-Intelligence Traders (ZIT)** and **Reinforcement Learning (RL)** agents. We analyze how market composition (regulated by the parameter $\omega$) affects price stability and the emergence of market stylized facts.

## Key Areas

### 1. Market Composition ($\omega$) Analysis
[cite_start]We explored three distinct market regimes:
* [cite_start]**RL-Dominated ($\omega=0.1$):** High excess kurtosis (22.90) and extreme fat tails due to strategic herding.
* **Balanced Market ($\omega=0.5$):** Strongest volatility clustering as RL and ZIT agents interact dynamically.
* [cite_start]**ZIT-Dominated ($\omega=0.9$):** Stabilizing liquidity dampens extreme movements, behaving more like a random walk.

### 2. Sensitivity Analysis (RL Reward Function)
The RL agents operate on a reward function defined as:
$$R_{t} = \Delta PnL_{t} - \lambda_{inv} \cdot |inventory_{t}| - \lambda_{hold} \cdot \mathbb{I}_{\{action=hold\}}$$

* **Inventory Penalty ($\lambda_{inv}$):** Higher values force agents to actively manage positions, leading to mean-reverting behavior toward zero.
* **Holding Penalty ($\lambda_{hold}$):** Increasing this penalty forces higher market activity, amplifying volatility clustering.

### 3. Reproducing Stylized Facts
Our simulation successfully qualitatively reproduced two core stylized facts:
* [cite_start]**Fat Tails:** Achieved an excess kurtosis of $\approx 2.08$ through wider private-value spans ($S=2000$). 
* **Volatility Clustering:** Lag-1 ACF of squared returns reached $\approx 0.18$, very close to the empirical target of 0.20.

### 4. Alternative RL Specification
We designed a fundamental-value reward function that replaces $PnL$ optimization with a bonus for trading near the fundamental price ($P_f$) and a penalty for trading during high bid-ask spreads.

## Technical Environment
* **Language:** Python 3.x
* **Key Implementation:** `lob_simulation.py` using Q-learning for RL agent policy optimization.

## Contributors
* **Abhishek Kumar Singh**
* **Konstantinos Papadimitriou**
* **Rishav Saha**

## References
* Mizuta, T., & Yagi, I. (2025). *Financial Market Design by an Agent-Based Model*.
