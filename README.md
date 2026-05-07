# Nengo Cognitive Robot: Biologically Plausible Agent with Curiosity

A neurally-inspired agent built in Nengo/NEF that navigates a colored grid world, learns transition statistics, and exhibits curiosity-driven exploration.

## Architecture

### Perception Pipeline (Biologically Inspired Vision)
- RGB input with Gaussian noise (σ=0.01)
- Opponent color transformation (red-green, blue-yellow, luminance)
- 4D cortical color space
- Cosine similarity classification (threshold 0.4)

### Memory Systems
- **Short-term**: Leaky integrator (0.9 decay) for previous color
- **Transition detection**: Outer product → 25-dimensional space
- **Long-term**: Integrators (0.99 recurrence) for red-origin transitions

### Behavior
- **Reflexive**: Wall avoidance from 3 proximity sensors
- **Cognitive**: Curiosity bias toward less-familiar color transitions
  - Familiarity = count_red→X / total_red_transitions
  - Bias = -0.6 × familiarity (soft repulsion)
- **Arbitration**: Priority scaling based on wall distance (emergency > moderate > open)

## Results
- Stable color recognition under moderate noise
- Accurate transition counting for red-origin sequences
- Observable exploration diversification in open environments

## Technologies
- Nengo / NEF
- Python
- Neural population coding
- Dynamical systems

## Future Work
- Hopfield networks for context-sensitive memory
- Predictive processing / prediction error signals
- Multi-modal integration
- Long-term adaptation via synaptic plasticity

## Author
Chiara Benini
