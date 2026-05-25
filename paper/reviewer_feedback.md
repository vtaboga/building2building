<!-- -*- mode: markdown -*- -->

# Reviewer feedback — verbatim

Verbatim reviewer comments on the accepted RLJ/RLC submission, preserved
for the camera-ready pass and any follow-up work. The paper is accepted;
this file is a record of the substantive critiques that the camera-ready
prose should acknowledge (per `AGENTS.md`'s "non-trivial paper changes
need explicit approval" rule, the actual response prose goes in
`main.tex`, not here).

Source: extracted from `refactoring/experiment_plan` (a pre-cleanup
working file) during the 2026-05-12 repository purge.

---

## Reviewer comments

In theory, these claims are supported just by producing the proposed
benchmark (and writing a paper about it), and then having some train/test
split of environments/settings for the generalization tests. Thus, it's
difficult to ascertain to what level of support the paper provides for
the proposed contributions. It's also difficult to evaluate the "key
aspects of generalization in RL" — what are these key aspects? How do
these proposed problem settings map to these key aspects?

Some things that I noted that were unsupportive of these claims:

The paper provides several different problem settings — goal adaptation,
dynamics adaptation, action space transfer, and cross-domain
generalization. However, the submission provides experimentation on only
the cross-domain generalization. While this seemingly supports the
listed contribution of producing a benchmark that can "evaluate key
aspects of generalization in RL," it seems like there is missing support
for the inclusion of the other problem settings. Why is goal adaptation
included? Is that a form of generalization? Where are the experiments
for goal adaptation? These same questions can also be asked of the
dynamics adaptation and action space transfer problem settings.

For the proposed problem settings, listed in my previous point, there
is a lack of rationale for why these problems are included, and why the
greater RL community should focus on these problems. For example, the
dynamics adaptation problem setting includes three difficulty levels —
how are these difficulties defined? Are they based on sample efficiency
of some agent? What makes one more difficult from another? For the
action-space transfer setting, is it reasonable to expect an RL agent
trained on an action space of size 5 to generalize to an action space
of size 10? How are the similarities/differences between domains in the
cross-domain generalization setting measured? What does "similar" mean
compared to "slightly different"?

In many papers that propose benchmarks, it is common for the authors to
include a comprehensive set of benchmark results across a wide swath of
algorithms [1] [2] [3]. However, in this work, the only included
algorithms are PPO and a logic controller. And with the proposal of a
benchmark aimed at generalization, it seems odd to not include any
algorithms that are designed to handle this problem setting. Perhaps
meta-RL algorithms like MAML [4], RL^2 [5], Amago-2 [6], or VariBad [7],
can handle this type of generalization. In the goal adaptation setting,
it is widely noted in MTRL that the scale of rewards is extremely
important for optimizing a single network to perform multiple tasks [8].
Is the problem setting in this submission different? Or would
normalizing the rewards be effective in enabling goal generalization?
Recent work, given that rewards are normalized/designed correctly, have
even found that MTRL can scale fairly well [9], [10]. Why are those
algorithms not evaluated on this proposed benchmark? Wouldn't it be
interesting to know if scale can't solve every task? The ProcGen [11]
benchmark noted that it seemed that larger models generalized better —
wouldn't it be interesting to know if that trend continued in a new
problem setting?

This comprehensive set of benchmark results would also help the
community, when reading this submission, have a frame of reference for
the performance profiles of these algorithms.

It is unclear from the current text how the hyperparameters of the
algorithms (PPO at least) are tuned. An interesting thought I had while
reading this submitted manuscript is that one potentially useful method
of hyperparameter tuning could be the Cross-environment Hyperparameter
Tuning method from RLC 2024 [12].

### Empirical Results Discussion

The results of the paper are missing a lot of the technical information
needed to replicate their results — number of seeds, measure of
statistical confidence, how hyperparameters were tuned, experiment
details. The paper, as noted in the discussion of unsupported claims,
is definitely missing common MTRL and generalization in RL (I noted
meta-RL algorithms above, but this could be more general RL algorithms
for generalization) algorithms.

---

## Cited references

1. Reginald McLean, Evangelos Chatzaroulas, Luc McCutcheon, Frank Roder,
   Tianhe Yu, Zhanpeng He, K.R. Zentner, Ryan Julian, J K Terry, Isaac
   Woungang, Nariman Farsad, & Pablo Samuel Castro (2025). Meta-World+:
   An Improved, Standardized, RL Benchmark. *NeurIPS Datasets and
   Benchmarks Track.*
2. Bellemare, M., Naddaf, Y., Veness, J., & Bowling, M. (2013). The
   arcade learning environment: an evaluation platform for general
   agents. *J. Artif. Int. Res.*, 47(1), 253–279.
3. Tongzhou Mu et al. (2021). ManiSkill: Generalizable Manipulation
   Skill Benchmark with Large-Scale Demonstrations. *NeurIPS Datasets
   and Benchmarks Track (Round 2).*
4. Chelsea Finn, Pieter Abbeel, & Sergey Levine (2017). Model-Agnostic
   Meta-Learning for Fast Adaptation of Deep Networks. *ICML.*
5. Yan Duan et al. (2017). RL^2: Fast Reinforcement Learning via Slow
   Reinforcement Learning.
6. Jake Grigsby et al. (2024). AMAGO-2: Breaking the Multi-Task Barrier
   in Meta-Reinforcement Learning with Transformers. *NeurIPS.*
7. Zintgraf, L. et al. (2021). VariBAD: variational Bayes-adaptive deep
   RL via meta-learning. *JMLR*, 22(1).
8. Hessel, M. et al. (2019). Multi-task deep reinforcement learning
   with PopArt. *AAAI.*
9. Reginald McLean et al. (2025). Multi-Task Reinforcement Learning
   Enables Parameter Scaling. *Reinforcement Learning Journal*, vol. 6,
   pp. 1075–1093.
10. Michal Nauman et al. (2025). Bigger, Regularized, Categorical:
    High-Capacity Value Functions are Efficient Multi-Task Learners.
    *NeurIPS.*
11. Cobbe, K., Hesse, C., Hilton, J., & Schulman, J. (2020). Leveraging
    procedural generation to benchmark reinforcement learning. *ICML.*
12. Andrew Patterson et al. (2024). Cross-environment Hyperparameter
    Tuning for Reinforcement Learning. *Reinforcement Learning
    Conference.*
