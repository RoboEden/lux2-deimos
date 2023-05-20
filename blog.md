# Deimos's RL approach

## Background
Our colleagues participated in the Lux Season 1 competition last year and conducted an in-depth study on the competition environment. During the preparation phase of this year's competition, we were invited by Stone to rewrite the Lux environment using JAX framework, which resulted in considerable speedup effect (overall user experience still needs improvement and we plan to optimize further in the future). Therefore, we also wanted to participate in Lux Season two to gain first-hand experience.

## Method
Our policy for this competition remains the utilization of the PPO algorithm, and the specific implementation follows the off-policy approach provided in RLlib. Since the observation space of this season's competition environment is structured similarly to that of the previous season, we have adopted the neural network architecture outlined in [1] for the backbone of our model and have made our code available as a reference in [2]. However, we have also made some modifications to our model that are unique to this season's competition environment, which we will highlight in the following sections.

### Feature space
The feature space consisted of three main components: global information, map features, and action queue. Each component was designed to capture critical information that allowed our robot to operate efficiently in the given environment.

- Global Information
The first component of our feature space is global information, which includes the current time phase of the environment (step, cycle, hour) and the total number of robots and resources. We utilized one-hot encoding to represent certain features, resulting in a feature dimension of 32.

- Map Features
The map feature component contained information about the resources (lichen, ice, ore, rubble) present in each block as well as the robot (unit, factory) located in each block. The feature dimension was designed to be 48x48x30.

- Action Queue
The last component of our feature space was the action queue, which represented the current set of actions in robots' queues. This component had a feature dimension of 48x48x20x6.

To encode our feature space, we then replicated this encoded global information for each grid. We used ResNet to merge the three components to achieve maximum efficiency. 

### Action related

The biggest challenge for RL players was managing the massive action space in the second season. To address this, we restrict the number of actions sent to a robot to one per turn, i.e., we completely ignore the action queue, and turn it into a real-time control problem. Unlike flg's approach, we allow the action to be repeated, thus some power can be saved. Besides, we implement a relatively complicated conditional sampling as follows.

#### Sampling Order for action sequences 

<!-- In the PPO algorithm, the neural network outputs a probability distribution for actions, and actions are then sampled from this distribution. Consequently, in typical game environments, the output dimension of the PPO actor head is equivalent to the size of the action space. For a robot of the unit type, the action space is composed of several combined features, including: -->
We sample different dimensions of action in the following order.
- Action type (move, transfer, pick up, dig, self-destruct, recharge, do nothing)
- Action direction (5 directions)
- Resource type for action (ice, ore, water, metal, power)
- Quantity of resources for action (float, 0-1)
- Infinite repeat or not (boolean)
- n (integer)

The dimensions sampled before affect the dimensions sampled later, so the whole action spaced is modeled as a conditional probability distribution.

$$\begin{align*}
& P(\text{type, direction, resource, quantity, repeat, infinite}) \\ 
= & P(\text{type}) \times  P(\text{direction | type}) \times  P(\text{resource | type, direction}) \times \\
& P(\text{quantity | type, direction, resource}) \times \\
& P(\text{repeat | type, direction, resource, quantity}) \times \\
& P(\text{n | type, direction, resource, quantity, repeat})
\end{align*}$$

#### Continuous Value Sampling

From the action space of the unit type robot described above, we can observe that the quantity of resources operated is a ranged real value, such as $[0,1]$. Generally, for actions involving continuous numerical values, researchers usually model them as a Gaussian distribution, but the gaussian distribution is unbounded, so some truncation is needed. Here, we adopted a more reasonable distribution, the beta distribution, which falls into $[0,1]$ by nature. The advantages of this approach can be found in our reference at [3].

## Model Architecture

In brief, is an 8-layer-deep 64-neuron-wide Resnet with some additional feature pre-processing layer and action/value head. The backbone is the same as [1], and the specific implementation can be found in [2]. 


## Training && Performance

### Random Initialization
The distributed reinforcement learning framework consists of two types of processes:
 - Rollout worker : interact with the environment to generate data, and 
 - Trainer :  perform policy optimization.

As this competition is a perfect information game, we set Rollout worker to play against itself, i.e., self-play. However, we found that the model did not explore new game states during the training process. Therefore, we downloaded a large number of replays from the Kaggle, randomly selected various game state from them, and used these intermediate states as the initial settings for the self-play process. This helped our strategy to explore many game situations that were difficult to evolve through self-play, and the data download can be referenced via the Lux season one sharing blog.

### CURRICULUM LEARNING PHASES

There are three main phases during the training process.
1. dense reward + rule-based factory spawning
2. dense reward + neural network factory spawning
3. sparse reward + neural network factory spawning

#### Phase1
The reward is set to dense reward as shown in the table below, and in this phase, we wrote a script for choosing factory spawning location and initial water/metal. In addition, if a player loses the game, it will be penalized by a large negative reward, which cancel out all its past reward. Thus, only the winner get positive reward in this phase.

| Reward type | value |
| -------- | -------- |
| reward for victory | 0.1 |
| penalty for defeat | -0.1 |
| light units increment | 0.04 |
| heavy units increment | 0.4 |
| ice increment| 0.0005 |
| ore increment | 0.001 |
| water increment | 0.001 |
| metal increment | 0.002 |
| power increment | 0.00005 |
| growing lichen  | 0.001 |
| penalty for losing factory| -0.5 |
| reward for surviving in this round | 0.01 |

Most of the reward can also be a penalty. For example, if a player builds a light unit, it will get a reward of 0.04, but if it loses the light unit, it will get a penalty of -0.04.

#### Phase2
The dense reward as described above is still used in this phase, but the factory spawning is now decided by the neural network. This phase is quite short. As long as robots know how to collect resources during phase 1, neural networks can learn to spawn factories in a few hours.

#### Phase3
In this phase, we switch to a sparse reward.
| Reward type | value |
| -------- | -------- |
| reward for lichen increment | 0.001 |
| reward for surviving in this round | 0.01 |

During this phase, the reward for lichen increment is zero-sum. All lichen reward a player gain will also become a penalty for its opponent, so the goal for a player becomes to gain more lichen than its opponent.

# Conclusion
Our training gets a smooth start, and we can see the ELO score increase steadily. However, it was stuck around 1900 in the last two month. We did lots of experiments to improve the ELO score, but none of them worked. In general, our approach is quite similar to flg's. Both ours and flg's follows last year's champion's approach, but in detail, we have different action space modeling, different rewards, and different curriculum learning phases. Maybe there is a bug; maybe our action space modeling is too complicated; maybe our reward is not good enough. We can't figure out the reason, because RL system is too complex to debug. It consists of too many components, and each component has too many hyper-parameters. Most time, if you did wrong in one component, you just get a bad ELO score silently without any explicit exceptions. Despite all the above, we still want to share our approach with you. 

## References
```
[1]  Chen, H., Tao, S., Chen, J., Shen, W., Li, X., Yu, C., Cheng, S., Zhu, X., & Li, X. (2023). Emergent collective intelligence from massive-agent cooperation and competition. arXiv preprint [arXiv:2301.01609](https://arxiv.org/pdf/2301.01609.pdf).

[2] https://github.com/RoboEden/lux2-deimos

[3] Po-Wei Chou, Daniel Maturana, Sebastian Scherer Proceedings of the 34th International Conference on Machine Learning, [PMLR 70:834-843, 2017](https://proceedings.mlr.press/v70/chou17a.html).
```
