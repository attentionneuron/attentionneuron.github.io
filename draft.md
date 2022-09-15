
## Abstract 

In complex systems, we often observe complex global behavior emerge from a collection of agents interacting with each other in their environment, with each individual agent acting only on locally available information, without knowing the full picture. Such systems have inspired development of artificial intelligence algorithms in areas such as swarm optimization and cellular automata. Motivated by the emergence of collective behavior from complex cellular systems, we build systems that feed each sensory input from the environment into distinct, but identical neural networks, each with no fixed relationship with one another. We show that these sensory networks can be trained to integrate information received locally, and through communication via an attention mechanism, can collectively produce a globally coherent policy. Moreover, the system can still perform its task even if the ordering of its inputs is randomly permuted several times during an episode.
These permutation invariant systems also display useful robustness and generalization properties that are broadly applicable.

______

## Introduction

Sensory substitution refers to the brain's ability to use one sensory modality (e.g., touch) to supply environmental information normally gathered by another sense (e.g., vision). Numerous studies have demonstrated that humans can adapt to changes in sensory inputs, even when they are fed into the *wrong* channels <dt-cite key="bach1969vision,bach2003sensory,sandlin2019backwards,eagleman2020livewired"></dt-cite>.
But difficult adaptations--such as learning to “see” by interpreting visual information emitted from a grid of electrodes placed on one's tongue <dt-cite key="bach2003sensory"></dt-cite>, or learning to ride a “backwards” bicycle <dt-cite key="sandlin2019backwards"></dt-cite>--require months of training to attain mastery.
Can we do better, and create artificial systems that can rapidly adapt to sensory substitutions, without the need to be retrained?

<div style="text-align: left;">
<figcaption style="color:#FF6C00;">Interactive Demo</figcaption><br/>
<div id="intro_demo" class="unselectable" style="text-align: left;"></div>
<figcaption style="text-align: left;">
<b>Permutation Invariant Cart-Pole Swing Up Demo</b><br/>
A permutation invariant network performing <i>CartpoleSwingupHarder</i>. Shuffle the order of the 5 observations at any time, and see how the agent adapts to the new ordering of the observations.
</figcaption>
</div>

Modern deep learning systems are generally unable to adapt to a sudden reordering of sensory inputs, unless the model is retrained, or if the user manually corrects the ordering of the inputs for the model. However, techniques from continual meta-learning, such as adaptive weights <dt-cite key="schmidhuber1992learning,ba2016using,ha2016hypernetworks"></dt-cite>, Hebbian-learning <dt-cite key="miconi2018differentiable,miconi2020backpropamine,najarro2020meta"></dt-cite>, and model-based <dt-cite key="deisenroth2011pilco,amos2018differentiable,ha2018worldmodels,hafner2018planet"></dt-cite> approaches can help the model adapt to such changes, and remain a promising active area of research.

In this work, we investigate agents that are explicitly designed to deal with sudden random reordering of their sensory inputs while performing a task. Motivated by recent developments in self-organizing neural networks <dt-cite key="fortuin2018som,mordvintsev2020growing,randazzo2020selfclassifying"></dt-cite> related to cellular automata <dt-cite key="neumann1966theory,codd2014cellular,conway1970game,wolfram1984cellular,chopard1998cellular"></dt-cite>, in our experiments, we feed each sensory input (which could be an individual state from a continuous control environment, or a patch of pixels from a visual environment) into an individual neural network module that integrates information from only this particular sensory input channel over time. While receiving information locally, each of these individual sensory neural network modules also continually broadcasts an output message. Inspired by the Set Transformer <dt-cite key="vaswani2017,set2019"></dt-cite> architecture, an attention mechanism combines these messages to form a global latent code which is then converted into the agent's action space. The attention mechanism can be viewed as a form of adaptive weights of a neural network, and in this context, allows for an arbitrary number of sensory inputs that can be processed in any random order.

In our experiments, we find that each individual sensory neural network module, despite receiving only localized information, can still collectively produce a globally coherent policy, and that such a system can be trained to perform tasks in several popular reinforcement learning (RL) environments. Furthermore, our system can utilize a varying number of sensory input channels in any randomly permuted order, even when the order is shuffled again several times during an episode.

<div style="text-align: left;">
<video class="b-lazy" data-src="assets/mp4/pong_occluded_reshuffle.mp4" type="video/mp4" autoplay muted playsinline loop style="margin: 0; width: 100%;" ></video>
<figcaption style="text-align: left;">
Our pong agent continues to work even when it is given a small subset (30%) of the screen, in a shuffled order. The screen is reshuffled multiple times during the game. For comparison, the actual game is shown on the left.
</figcaption>
</div>

Permutation invariant systems have several advantages over traditional fixed-input systems.
We find that encouraging a system to learn a coherent representation of a permutation invariant observation space leads to policies that are more robust and generalizes better to unseen situations.
We show that, without additional training, our system continues to function even when we inject additional input channels containing noise or redundant information.
In visual environments, we show that our system can be trained to perform a task even if it is given only a small fraction of randomly chosen patches from the screen, and at test time, if given more patches, the system can take advantage of the additional information to perform better.
We also demonstrate that our system can generalize to visual environments with novel background images, despite training on a single fixed background.
Lastly, to make training more practical, we propose a behavioral cloning scheme to convert policies trained with existing methods into a permutation invariant policy with desirable properties.

<div style="text-align: left;">
<video class="b-lazy" data-src="assets/mp4/yosemite.mp4" type="video/mp4" autoplay muted playsinline loop style="width:100%;" ></video>
<figcaption style="text-align: left;">
We find that a side-effect of permutation invariant RL agents is that without any additional training or fine-tuning, they also tend to work even when the original training background is replaced with various images.<br/>
</figcaption>
</div>

______

## Method

### Background

Our goal is to devise an agent that is permutation invariant (PI) in the action space to the permutations in the input space.
While it is possible to acquire a quasi-PI agent by training with randomly shuffled observations and hope the agent's policy network has enough capacity to memorize all the patterns, we aim for a design that achieves true PI even if the agent is trained with fix-ordered observations. Mathematically, we are looking for a non-trivial function $f(x): \mathcal{R}^n \mapsto \mathcal{R}^m$ such that $f(x[{s}]) = f(x)$ for any $x \in \mathcal{R}^n$, and $s$ is any permutation of the indices $\{1, \cdots, n\}$.
A different but closely related concept is permutation equivariance (PE) which can be described by a function $h(x): \mathcal{R}^n \mapsto \mathcal{R}^n$ such that $h(x[{s}]) = h(x)[s]$. Unlike PI, the dimensions of the input and the output must equal in PE.

Self-attentions can be PE. In its simplest form, self-attention is described as $y = \sigma(QK^{\top})V$ where $Q,K \in \mathcal{R}^{n \times d_q}, V \in \mathcal{R}^{n \times d_v}$ are the Query, Key and Value matrices and $\sigma(\cdot)$ is a non-linear function. In most scenarios, $Q, K, V$ are functions of the input $x \in \mathcal{R}^n$ (e.g. linear transformations), and permuting $x$ therefore is equivalent to permuting the rows in $Q, K, V$ and based on its definition it is straightforward to verify the PE property. Set Transformer <dt-cite key="set2019"></dt-cite> cleverly replaced $Q$ with a set of learnable seed vectors, so it is no longer a function of input $x$, thus enabling the output to become PI.

Here, we provide a simple, non-rigorous example demonstrating permutation invariant property of the self-attention mechanism, to give some intuition to readers who may not be familiar with self-attention. For a detailed treatment, please refer to <dt-cite key="zaheer2017deep,set2019"></dt-cite>.

As mentioned earlier, in its simplest form, self-attention is described as:

&nbsp;&nbsp;&nbsp;&nbsp;$y = \sigma(QK^{\top})V$
<!--<div style="text-align: center;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/equation_pi_explanation_part_0.png" style="display: block; margin: auto; width: 100%;"/>
</div>-->

where $Q \in \mathcal{R}^{N_q \times d_q}, K \in \mathcal{R}^{N \times d_q}, V \in \mathcal{R}^{N \times d_v}$ are the Query, Key and Value matrices and $\sigma(\cdot)$ is a non-linear function. In this work, $Q$ is a fixed matrix, and $K, V$ are functions of the input $X \in \mathcal{R}^{N \times d_{in}}$ where $N$ is the number of observation components (equivalent to the number of sensory neurons) and $d_{in}$ is the dimension of each component. In most settings, $K=X W_k, V=X W_v$ are linear transformations, thus permuting $X$ therefore is equivalent to permuting the rows in $K, V$.

We would like to show that the output $y$ is the same regardless of the ordering of the rows of $K, V$. For simplicity, suppose $N=3$, $N_q=2$, $d_q=d_v=1$, so that $Q \in \mathcal{R}^{2 \times 1}$, $K \in \mathcal{R}^{3 \times 1}$, $V \in \mathcal{R}^{3 \times 1}$:

<div style="text-align: center;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/equation_pi_explanation_part_1.larger.png" style="display: block; margin: auto; width: 100%;"/>
</div>

The output $y \in \mathcal{R}^{2 \times 1}$ remains the same when the rows of $K, V$ are permuted from $[1, 2, 3]$ to $[3, 1, 2]$:

<div style="text-align: center;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/equation_pi_explanation_part_2.larger.png" style="display: block; margin: auto; width: 100%;"/>
</div>

We have highlighted the same terms with the same color in both equations to show the results are indeed identical. In general, we have $y_{ij} = \sum_{b=1}^{N} \sigma [ \sum_{a=1}^{d_q} Q_{ia} K_{ba} ] V_{bj}$. Permuting the input is equivalent to permuting the indices $b$ (i.e. rows of $K$ and $V$), which only affects the order of the outer summation and does not affect $y_{ij}$ because summation is a permutation invariant operation. Notice that in the above example and the proof here we have assumed that $\sigma(\cdot)$ is an element-wise operation--a valid assumption since most activation functions satisfy this condition.<dt-fn>Applying <i>softmax</i> to each row only brings scalar multipliers to each row and the proof still holds.</dt-fn>

As we'll discuss next, this formulation lets us convert an observation signal from the RL environment into a permutation invariant representation $y$. We'll use this representation in place of the actual observation as the input that goes into the downstream policy network of an RL agent.

### Sensory Neurons with Attention

To create permutation invariant (PI) agents, we propose to add an extra layer in front of the agent's policy network $\pi$, which accepts the current observation $o_t$ and the previous action $a_{t-1}$ as its inputs. We call this new layer AttentionNeuron, and the following figure gives an overview of our method:

<div style="text-align: center;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/attentionneuron.png" style="display: block; margin: auto; width: 100%;"/>
<figcaption style="text-align: left;">
<b>Overview of Method</b><br/>
AttentionNeuron is a standalone layer, in which each sensory neuron only has access to a part of the unordered observations <span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>o</mi><mi>t</mi></msub></mrow><annotation encoding="application/x-tex">o_t</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.43056em;"></span><span class="strut bottom" style="height:0.58056em;vertical-align:-0.15em;"></span><span class="base textstyle uncramped"><span class="mord"><span class="mord mathit">o</span><span class="vlist"><span style="top:0.15em;margin-right:0.05em;margin-left:0em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle cramped"><span class="mord mathit">t</span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span></span></span></span>. Together with the agent's previous action <span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>a</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub></mrow><annotation encoding="application/x-tex">a_{t-1}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.43056em;"></span><span class="strut bottom" style="height:0.638891em;vertical-align:-0.208331em;"></span><span class="base textstyle uncramped"><span class="mord"><span class="mord mathit">a</span><span class="vlist"><span style="top:0.15em;margin-right:0.05em;margin-left:0em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle cramped"><span class="mord scriptstyle cramped"><span class="mord mathit">t</span><span class="mbin">−</span><span class="mord mathrm">1</span></span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span></span></span></span>, each neuron generates messages independently using the shared functions <span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>f</mi><mi>k</mi></msub><mo>(</mo><msub><mi>o</mi><mi>t</mi></msub><mo>[</mo><mi>i</mi><mo>]</mo><mo separator="true">,</mo><msub><mi>a</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub><mo>)</mo></mrow><annotation encoding="application/x-tex">f_k(o_t[i], a_{t-1})</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.75em;"></span><span class="strut bottom" style="height:1em;vertical-align:-0.25em;"></span><span class="base textstyle uncramped"><span class="mord"><span class="mord mathit" style="margin-right:0.10764em;">f</span><span class="vlist"><span style="top:0.15em;margin-right:0.05em;margin-left:-0.10764em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle cramped"><span class="mord mathit" style="margin-right:0.03148em;">k</span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span><span class="mopen">(</span><span class="mord"><span class="mord mathit">o</span><span class="vlist"><span style="top:0.15em;margin-right:0.05em;margin-left:0em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle cramped"><span class="mord mathit">t</span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span><span class="mopen">[</span><span class="mord mathit">i</span><span class="mclose">]</span><span class="mpunct">,</span><span class="mord"><span class="mord mathit">a</span><span class="vlist"><span style="top:0.15em;margin-right:0.05em;margin-left:0em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle cramped"><span class="mord scriptstyle cramped"><span class="mord mathit">t</span><span class="mbin">−</span><span class="mord mathrm">1</span></span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span><span class="mclose">)</span></span></span></span> and <span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>f</mi><mi>v</mi></msub><mo>(</mo><msub><mi>o</mi><mi>t</mi></msub><mo>[</mo><mi>i</mi><mo>]</mo><mo>)</mo></mrow><annotation encoding="application/x-tex">f_v(o_t[i])</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.75em;"></span><span class="strut bottom" style="height:1em;vertical-align:-0.25em;"></span><span class="base textstyle uncramped"><span class="mord"><span class="mord mathit" style="margin-right:0.10764em;">f</span><span class="vlist"><span style="top:0.15em;margin-right:0.05em;margin-left:-0.10764em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle cramped"><span class="mord mathit" style="margin-right:0.03588em;">v</span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span><span class="mopen">(</span><span class="mord"><span class="mord mathit">o</span><span class="vlist"><span style="top:0.15em;margin-right:0.05em;margin-left:0em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle cramped"><span class="mord mathit">t</span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span><span class="mopen">[</span><span class="mord mathit">i</span><span class="mclose">]</span><span class="mclose">)</span></span></span></span>. The attention mechanism summarizes the messages into a global latent code <span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>m</mi><mi>t</mi></msub></mrow><annotation encoding="application/x-tex">m_t</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.43056em;"></span><span class="strut bottom" style="height:0.58056em;vertical-align:-0.15em;"></span><span class="base textstyle uncramped"><span class="mord"><span class="mord mathit">m</span><span class="vlist"><span style="top:0.15em;margin-right:0.05em;margin-left:0em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle cramped"><span class="mord mathit">t</span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span></span></span></span>.
</figcaption>
</div>
<!--actual caption in markdown, since it doens't work in the figure caption.-->
<!--AttentionNeuron is a standalone layer, in which each sensory neuron only has access to a part of the unordered observations $o_t$. Together with the agent's previous action $a_{t-1}$, each neuron generates messages independently using the shared functions $f_k(o_t[i], a_{t-1})$ and $f_v(o_t[i])$. The attention mechanism summarizes the messages into a global latent code $m_t$.-->

The operations inside AttentionNeuron can be described by the following two equations:

<div style="text-align: center;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/attentionneuron_equations.larger.png" style="display: block; margin: auto; width: 100%;"/>
</div>

Equation 1 shows how each of the $N$ sensory neuron independently generates its messages $f_k$ and $f_v$, which are functions shared across all sensory neurons. Equation 2 shows the attention mechanism aggregate these messages. Note that although we could have absorbed the projection matrices $W_q, W_k, W_v$ into $Q, K, V$, we keep them in the equation to show explicitly the formulation. Equation 2 is almost identical to the simple definition of self-attention mentioned earlier. Following <dt-cite key="set2019"></dt-cite>, we make our $Q$ matrix a bank of fixed embeddings, rather than depend on the observation $o_t$.

Note that permuting the observations only affects the row orders of $K$ and $V$, and that applying the same permutation to the rows of both $K$ and $V$ still results in the same $m_t$ which is PI. 
As long as we set constant the number of rows in $Q$, the change in the input size affects only the number of rows in $K$ and $V$ and does not affect the output $m_t$. In other words, our agent can accept inputs of arbitrary length and output a fixed sized $m_t$. Later, we apply this flexibility of input dimensions to RL agents.

For clarity, the following table summarizes the notations as well as the corresponding setups we used for the experiments:

<div style="text-align: center;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/table_notation.png" style="display: block; margin: auto; width: 100%;"/>
<figcaption style="text-align: left;">
<b>Notation list</b><br/>
In this table, we also provide the dimensions used in our model for different RL environments, to give the reader a sense of the relative magnitudes involved in each part of the system.
</figcaption>
</div>

### Design Choices

It is worthwhile to have a discussion on the design choices made.
Since the ordering of the input is arbitrary, each sensory neuron is required to interpret and identify their received signal.
To achieve this, we want $f_k(o_t[i], a_{t-1})$ to have temporal memories.
In practice, we find both RNNs and feed-forward neural networks (FNN) with stacked observations work well, with FNNs being more practical for environments with high dimensional observations.

In addition to the temporal memory, including previous actions is important for the input identification too. Although the former allows the neurons to infer the input signals based on the characteristics of the temporal stream, this may not be sufficient. For example, when controlling a legged robot, most of the sensor readings are joint angles and velocities from the legs, which are not only numerically identically bounded but also change in similar patterns.
The inclusion of previous actions gives each sensory neuron a chance to infer the causal relationship between the input channel and the applied actions, which helps with the input identification.

Finally, in Equation 2 we could have combined $QW_q \in \mathcal{R}^{M \times d_q}$ as a single learnable parameters matrix, but we separate them for two reasons.
First, by factoring into two matrices, we can reduce the number of learnable parameters.
Second, we find that instead of making $Q$ learnable, using the positional encoding proposed in Transformer <dt-cite key="vaswani2017"></dt-cite> encourages the attention mechanism to generate distinct codes. Here we use the row indices in $Q$ as the positions for encoding.

______

## Experiments

We experiment on several different RL environments to study various properties of permutation invariant RL agents.
Due to the nature of the underlying tasks, we will describe the different architectures of the policy networks used and discuss various training methods.
However, the AttentionNeuron layers in all agents are similar, so we first describe the common setups.
Hyper-parameters and other details for all experiments are summarized in the Appendix.

For non-vision continuous control tasks, the agent receives an observation vector $o_t \in \mathcal{R}^{|O|}$ at time $t$. We assign $N=|O|$ sensory neurons for the tasks, each of which sees one element from the vector, hence $o_t[i] \in \mathcal{R}^1, i=1, \cdots, |O|$. We use an LSTM <dt-cite key="lstm1997"></dt-cite> as our $f_k(o_t[i], a_{t-1})$ to generate Keys, the input size of which is $1 + |A|$ ($2$ for Cart-Pole and $9$ for PyBullet Ant). A simple pass-through function $f(x) = x$ serves as our $f_v(o_t[i])$, and $\sigma(\cdot)$ is $tanh$. For simplicity, we find $W_v = I$ works well for the tasks, so the learnable components are the LSTM, $W_q$ and $W_k$.

For vision based tasks, we gray-scale and stack $k=4$ consecutive RGB frames from the environment, and thus our agent observes $o_t \in \mathcal{R}^{H \times W \times k}$.
$o_t$ is split into non-overlapping patches of size $P=6$ using a sliding window, so each sensory neuron observes $o_t[i] \in \mathcal{R}^{6 \times 6 \times k}$.
Here, $f_v(o_t[i])$ flattens the data and returns it, hence $V(o_t)$ returns a tensor of shape $N \times d_{f_v} = N \times (6 \times 6 \times 4) = N \times 144$. Due to the high dimensionality for vision tasks, we do not use RNNs for $f_k$, but instead use a simpler method to process each sensory input. $f_k(o_t[i], a_{t-1})$ takes the difference between consecutive frames ($o_t[i]$), then flattens the result, appends $a_{t-1}$, and returns the concatenated vector. $K(o_t, a_{t-1})$ thus gives a tensor of shape $N \times d_{f_k}$ $=$ $N \times [(6 \times 6 \times 3) + |A|]$ $=$ $N \times (108 + |A|)$ (111 for CarRacing and 114 for Atari Pong). We use *softmax* as the non-linear activation function $\sigma(\cdot)$, and we apply layer normalization <dt-cite key="ba2016layer"></dt-cite> to both the input patches and the output latent code.

______

## Cart-pole swing up

We examine Cart-pole swing up <dt-cite key="Gal2016Improving,deepPILCOgithub,ha2017evolving,wann2019"></dt-cite> to first illustrate our method, and also use it to provide a clear analysis of the attention mechanism.
We use *CartPoleSwingUpHarder* <dt-cite key="learningtopredict2019"></dt-cite>, a more difficult version of the task where the initial positions and velocities are highly randomized, leading to a higher variance of task scenarios.
In the environment, the agent observes $[x, \dot{x}, cos(\theta), sin(\theta), \dot{\theta}]$, outputs a scalar action, and is rewarded at each step for getting $x$ close to 0 and $cos(\theta)$ close to 1.

<div style="text-align: left;">
<figcaption style="color:#FF6C00;">Interactive Demo</figcaption><br/>
<div id="cartpole_demo" class="unselectable" style="text-align: left;"></div>
<figcaption style="text-align: left;">
<b>Permutation Invariant Agent in CartPoleSwingUpHarder</b><br/>
In this demo, the user can shuffle the order of the 5 inputs at any time, and observe how the agent adapts to the new ordering of the inputs.
</figcaption>
</div>

We use a two-layer neural network as our agent. The first layer is an AttentionNeuron layer with $N=5$ sensory neurons and outputs $m_t \in \mathcal{R}^{16}$. A linear layer takes $m_t$ as input and outputs a scalar action. For comparison, we also trained an agent with a two-layer FNN policy with $16$ hidden units. We use direct policy search to train agents with CMA-ES <dt-cite key="hansen2006cma"></dt-cite>, an evolution strategies (ES) method.

We report experimental results in the following table:

<div style="text-align: center;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/table_cartpole_results.larger.png" style="display: block; margin: auto; width: 100%;"/>
<figcaption style="text-align: left;">
<b>Cart-pole Tests</b><br/>
For each experiment, we report the average score and the standard deviation from 1000 test episodes. Our agent is trained only in the environment with 5 sensory inputs.
</figcaption>
</div>

Our agent can perform the task and balance the cart-pole from an initially random state.
Its average score is slightly lower than the baseline (See column 1) because each sensory neuron requires some time steps in each episode to interpret the sensory input signal it receives. However, as a trade-off for the performance sacrifice, our agent can retain its performance even when the input sensor array is randomly shuffled, which is not the case for an FNN policy (column 2).
Moreover, although our agent is only trained in an environment with five inputs, it can accept an arbitrary number of inputs in any order without re-training.<dt-fn>Because our agent was not trained with normalization layers, we scaled the output from the AttentionNeuron layer by 0.5 to account for the extra inputs in the last 2 experiments.</dt-fn> We test our agent by duplicating the 5 inputs to give the agent 10 observations (column 3).
When we replace the 5 extra signals with white noises with $\sigma=0.1$ (column 4), we do not see a significant drop in performance.

The AttentionNeuron layer should possess 2 properties to attain these: its output is permutation invariant to its input, and its output carries task-relevant information.
The following figure is a visual confirmation of the permutation invariant property, whereby we plot the output messages from the layer and their changes over time from two tests. Using the same environment seed, we keep the observation as-is in the first test but we shuffle the order in the second. As the figure shows, the output messages are identical in the two roll-outs.

<div style="text-align: left;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/figure_cartpole_shuffle.png" style="display: block; margin: auto; width: 75%;"/>
<figcaption style="text-align: left;">
<b>Permutation invariant outputs</b><br/>
The output (16-dimensional global latent code) from the AttentionNeuron layer does not change when we input the sensor array as-is (top) or when we randomly shuffle the array (bottom). Yellow represents higher values, and blue for lower values.
</figcaption>
</div>

We also perform a simple linear regression analysis on the outputs (based on the shuffled inputs) to recover the 5 inputs in their original order.
The following table shows the $R^2$ values<dt-fn>$R^2$ measures the goodness-of-fit of a model. An $R^2$ of 1 implies that the regression perfectly fits the data.</dt-fn> from this analysis, suggesting that some important indicators (e.g. $\dot{x}$ and $\dot{\theta}$) are well represented in the output:

<div style="text-align: center;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/table_cartpole_explanation.larger.png" style="display: block; margin: auto; width: 100%;"/>
<figcaption style="text-align: left;">
<b>Linear regression analysis on the output</b><br/>
For each of the <span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>N</mi><mo>=</mo><mn>5</mn></mrow><annotation encoding="application/x-tex">N=5</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.68333em;"></span><span class="strut bottom" style="height:0.68333em;vertical-align:0em;"></span><span class="base textstyle uncramped"><span class="mord mathit" style="margin-right:0.10903em;">N</span><span class="mrel">=</span><span class="mord mathrm">5</span></span></span></span> sensory inputs we have one LR model with <span class="katex"><span class="katex-mathml"><math><semantics><mrow><msub><mi>m</mi><mi>t</mi></msub><mo>∈</mo><msup><mrow><mi mathvariant="script">R</mi></mrow><mrow><mn>1</mn><mn>6</mn></mrow></msup></mrow><annotation encoding="application/x-tex">m_t \in \mathcal{R}^{16}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="strut" style="height:0.8141079999999999em;"></span><span class="strut bottom" style="height:0.964108em;vertical-align:-0.15em;"></span><span class="base textstyle uncramped"><span class="mord"><span class="mord mathit">m</span><span class="vlist"><span style="top:0.15em;margin-right:0.05em;margin-left:0em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle cramped"><span class="mord mathit">t</span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span><span class="mrel">∈</span><span class=""><span class="mord textstyle uncramped"><span class="mord mathcal">R</span></span><span class="vlist"><span style="top:-0.363em;margin-right:0.05em;"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span><span class="reset-textstyle scriptstyle uncramped"><span class="mord scriptstyle uncramped"><span class="mord mathrm">1</span><span class="mord mathrm">6</span></span></span></span><span class="baseline-fix"><span class="fontsize-ensurer reset-size5 size5"><span style="font-size:0em;">​</span></span>​</span></span></span></span></span></span> as the explanatory variables.
</figcaption>
</div>
<!--For each of the $N=5$ sensory inputs we have one linear regression model with $m_t \in \mathcal{R}^{16}$ as the explanatory variables.-->

Finally, to accompany the quantitative results in this section, we extended the earlier interactive demo to showcase the flexibility of PI agents. Here, our agent, with no additional training, receives 15 input signals in shuffled order, ten of which are pure noise, and the other five are the actual observations from the environment.

<div style="text-align: left;">
<a name="noise-demo"></a>
<figcaption style="color:#FF6C00;">Interactive Demo</figcaption><br/>
<div id="cartpole_demo_special" class="unselectable" style="text-align: left;"></div>
<figcaption style="text-align: left;">
<b>Dealing with unspecified number of extra noisy channels</b><br/>
Without additional training, our agent receives 15 input signals in shuffled order, 10 of which are pure Gaussian noise (σ=0.1), and the other 5 are the actual observations from the environment. Like the earlier demo, the user can shuffle the order of the 15 inputs, and observe how the agent adapts to the new ordering of the inputs.
</figcaption>
</div>

The existing policy is still able to perform the task, demonstrating the system's capacity to work with a large number of inputs and attend only to channels it deems useful. Such flexibility may find useful applications for processing a large unspecified number of signals, most of which are noise, from ill-defined systems.

______

## PyBullet Ant

While direct policy search methods such as evolution strategies (ES) can train permutation invariant RL agents, oftentimes we already have access to pre-trained agents or recorded human data performing the task at hand.
Behavior cloning (BC) can allow us to convert an existing policy to a version that is permutation invariant with desirable properties associated with it. We report experimental results here:

<div style="text-align: left;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/table_bulletant_results.larger.png" style="display: block; margin: auto; width: 100%;"/>
<figcaption style="text-align: left;">
<b>PyBullet Ant Experimental Results</b>
</figcaption>
</div>

We train a standard two-layer FNN policy to perform *AntBulletEnv-v0*, a 3D locomotion task in PyBullet <dt-cite key="coumans2020"></dt-cite>, and use it as a teacher for BC. For comparison, we also train a two-layer agent with AttentionNeuron for its first layer. Both networks are trained with ES.
Similar to CartPole, we expect to see a small performance drop due to some time steps required for the agent to interpret an arbitrarily ordered observation space.
We then collect data from the FNN teacher policy to train permutation invariant agents using BC. More details of the BC setup can be found in the Appendix.

The performance of the BC agent is lower than the one trained from scratch with ES, despite having the identical architecture.
This suggests that the inductive bias that comes with permutation invariance may not match the original teacher network, so the small model used here may not be expressive enough to clone any teacher policy, resulting in a larger variance in performance. A benefit of gradient-based BC, compared to RL, is that we can easily train larger networks to fit the behavioral data. We show that increasing the size of the subsequent layers for BC does enhance the performance.

While not explicitly trained to do so, we note that the policy still works even when we reshuffle the ordering of the observations several times during an episode:

<div style="text-align: left;">
<video class="b-lazy" data-src="assets/mp4/ant.mp4" type="video/mp4" autoplay muted playsinline loop style="margin: 0; width: 100%;" ></video>
<figcaption style="text-align: left;">
PyBullet Ant with a permutation invariant policy.<br/>
The ordering of the 28 observations is reshuffled every 100 frames.<br/>
</figcaption>
</div>

As we will demonstrate next, BC is a useful technique for training permutation invariant agents in environments with high dimensional visual observations that may require larger networks.

______

## Atari Pong

Here, we are interested in solving screen-shuffled versions of vision-based RL environments, where each observation frame is divided up into a grid of patches, and like a puzzle, the agent must process the patches in a shuffled order to determine a course of action to take. A shuffled version of Atari Pong <dt-cite key="openai_gym"></dt-cite>, in the following figure, can be especially hard for humans to play when inductive biases from human priors <dt-cite key="dubey2018investigating"></dt-cite> that expect a certain type of spatial structure is missing from the observations:

<div style="text-align: left;">
<video class="b-lazy" data-src="assets/mp4/pong_reshuffle.mp4" type="video/mp4" autoplay muted playsinline loop style="margin: 0; width: 100%;" ></video>
<figcaption style="text-align: left;">
<b>Pong and <i>Puzzle Pong</i></b>
</figcaption>
</figcaption>
</div>

But rather than throwing away the spatial structure entirely from our solution, we find that convolution neural network (CNN) policies work better than fully connected multi-layer perceptron (MLP) policies when trained with behavior cloning for Atari Pong. In this experiment, we reshape the output $m_t$ of the AttentionNeuron layer from $\mathcal{R}^{400 \times 32}$ to $\mathcal{R}^{20 \times 20 \times 32}$, a 2D grid of latent codes, and pass this 2D grid into a CNN policy. This way, the role of the AttentionNeuron layer is to take a list of unordered observation patches, and learn to construct a 2D grid representation of the inputs to be used by a downstream policy that expects some form of spatial structure in the codes. Our permutation invariant policy trained with BC can consistently reach a perfect score of 21, even with shuffled screens. The details of the CNN policy and BC training can be found in the Appendix.

Unlike typical CNN policies, our agent can accept a subset of the screen, since the agent's input is a variable-length list of patches.
It would thus be interesting to deliberately randomly discard a certain percentage of the patches and see how the agent reacts.
The net effect of this experiment for humans is similar to being asked to play a partially occluded and shuffled version of Atari Pong. During training via BC, we randomly remove a percentage of observation patches. In tests, we fix the randomly selected positions of patches to discard during an entire episode. The following figure demonstrates the agent's effective policy even when we also remove 70% of the patches:

<div style="text-align: left;">
<video class="b-lazy" data-src="assets/mp4/pong_occluded_reshuffle.mp4" type="video/mp4" autoplay muted playsinline loop style="margin: 0; width: 100%;" ></video>
<figcaption style="text-align: left;">
70% Occluded, Shuffled-screen Atari Pong (right). Observations reshuffled every 500 frames.
</figcaption>
</div>

We present the results in a heat map in the following figure, where the y-axis shows the patches removed during training and the x-axis gives the patch occlusion ratio in tests:

<div style="text-align: left;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/pong_results.larger.png" style="display: block; margin: auto; width: 100%;"/>
<figcaption style="text-align: left;">
<b>Linear regression analysis on the output</b><br/>
Mean test scores in Atari Pong, and example of a randomly-shuffled occluded observation.} In the heat map, each value is the average score from 100 test episodes.
</figcaption>
</div>

The heat map shows clear patterns for interpretation.
Looking horizontally along each row, the performance drops because the agent sees less of the screen which increases the difficulty.
Interestingly, an agent trained at a high occlusion rate of $80\%$ rarely wins against the Atari opponent, but once it is presented with the full set of patches during tests, it is able to achieve a fair result by making use of the additional information.

To gain insights into understanding the policy, we projected the AttentionNeuron layer's output in a test roll-out to 2D space using t-SNE <dt-cite key="van2008visualizing"></dt-cite>. In the figure below, we highlight several groups and show their corresponding inputs. The AttentionNeuron layer clearly learned to cluster inputs that share similar features:

<div style="text-align: center;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/figure_pong_tsne.png" style="display: block; margin: auto; width: 100%;"/>
<figcaption style="text-align: left;">
<b>2D embedding of the AttentionNeuron layer's output in a test episode</b><br/>
We highlight several representative groups in the plot, and show the sampled inputs from them.
For each group, we show 3 corresponding inputs (rows) and unstack each to show the time dimension (columns). 
</figcaption>
</div>

For example, the 3 sampled inputs in the blue group show the situation when the agent's paddle moved toward the bottom of the screen and stayed there. Similarly, the orange group shows the cases when the ball was not in sight, this happened right before/after a game started/ended. We believe these discriminative outputs enabled the downstream policy to accomplish the agent's task.

______

## Car Racing

<div style="text-align: left;">
<video class="b-lazy" data-src="assets/mp4/car_racing.mp4" type="video/mp4" autoplay muted playsinline loop style="margin: 0; width: 100%;" ></video>
<figcaption style="text-align: left;">
<b>CarRacing base task (left), modified shuffled-screen task (right)</b><br/>
Our agent is only trained on this environment.
The right screen is what our agent observes and the left is for human visualization. A human will find driving with the shuffled observation to be very difficult because we are not constantly exposed to such tasks, just like in the “reverse bicycle” example mentioned earlier.
</figcaption>
</div>

We find that encouraging an agent to learn a coherent representation of a deliberately shuffled visual scene leads to agents with useful generalization properties.
Such agents are still able to perform their task even if the visual background of the environment changes, despite being trained only on a single static background.
Out-of-domain generalization is an active area, and here, we combine our method with AttentionAgent <dt-cite key="attentionagent2020"></dt-cite>, a method that uses selective, hard-attention via a patch voting mechanism. AttentionAgents in <dt-cite key="attentionagent2020"></dt-cite> generalize well to several unseen visual environments where task irrelevant elements are modified, but fail to generalize to drastic background changes in a zero-shot setting. We find that combining the permutation invariant AttentionNeuron layer with AttentionAgent's policy network results in good generalization performance when we change the background:

<div style="text-align: left;">
<video class="b-lazy" data-src="assets/mp4/kof.mp4" type="video/mp4" autoplay muted playsinline loop style="width:100%;" ></video>
<figcaption style="text-align: left;">
<b>KOF background</b>
</figcaption>
<video class="b-lazy" data-src="assets/mp4/mt_fuji.mp4" type="video/mp4" autoplay muted playsinline loop style="width:100%;" ></video>
<figcaption style="text-align: left;">
<b>Mt. Fuji background</b>
</figcaption>
<video class="b-lazy" data-src="assets/mp4/ds.mp4" type="video/mp4" autoplay muted playsinline loop style="width:100%;" ></video>
<figcaption style="text-align: left;">
<b>DS background</b>
</figcaption>
<video class="b-lazy" data-src="assets/mp4/ukiyoe.mp4" type="video/mp4" autoplay muted playsinline loop style="width:100%;" ></video>
<figcaption style="text-align: left;">
<b>Ukiyo-e background</b>
</figcaption>
</div>

As mentioned, we combine the AttentionNeuron layer with the policy network used in AttentionAgent. As the hard-attention-based policy is non-differentiable, we train the entire system using ES.
We reshape the AttentionNeuron layer's outputs to adapt for the policy network.
Specifically, we reshape the output message to $m_t \in \mathcal{R}^{32 \times 32 \times 16}$ such that it can be viewed as a 32-by-32 grid of 16 channels.
The end result is a policy with two layers of attention: the first layer outputs a latent code book to represent a shuffled scene, and the second layer performs hard attention to select the top $K=10$ codes from a 2D global latent code book. A detailed description of the selective hard attention policy from <dt-cite key="attentionagent2020"></dt-cite>, a method that uses selective, hard-attention via a patch voting mechanism. AttentionAgents in <dt-cite key="attentionagent2020"></dt-cite> and other training details can be found in the Appendix.

We first train the agent in the CarRacing <dt-cite key="carracing_v0"></dt-cite> environment, and report the average score from 100 test roll-outs in the following table.
As the first column shows, our agent's performance in the training environment is slightly lower but comparable to the baseline method, as expected. But because our agent accepts randomly shuffled inputs, it is still able to navigate even when the patches are shuffled.

<div style="text-align: left;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/table_carracing_results.larger.png" style="display: block; margin: auto; width: 100%;"/>
<figcaption style="text-align: left;">
<b>CarRacing Test Results</b>
</figcaption>
</div>


Without additional training or fine-tuning, we test whether the agent can also navigate in four modified environments where the green grass background is replaced with various images. In the CarRacing Test Result (from column 2) shows, our agent generalizes well to most of the test environments with only mild performance drops while the baseline method fails to generalize. We suspect this is because the AttentionNeuron layer has transformed the original RGB space to a useful hidden representation (represented by $m_t$) that has eliminated task irrelevant information after observing and reasoning about the sequences of $(o_t, a_{t-1})$ during training, enabling the downstream hard attention policy to work with an optimized abstract representation tailored for the policy, instead of raw RGB patches.

We also compare our method to NetRand <dt-cite key="lee2019network"></dt-cite>, a simple but effective technique developed to perform similar generalization tasks. In the second row of CarRacing Test Result Table are the results of training NetRand on the base CarRacing task. The CarRacing task proved to be too difficult for NetRand, but despite a low performance score of 480 in the training environment, the agent generalizes well to the “Mt. Fuji” and “Ukiyoe” modifications. In order to obtain a meaningful comparison, we combine NetRand with AttentionAgent so that it can get close to a mean score of 900 on the base task. To do that, we use NetRand as an input layer to the AttentionAgent policy network, and train the combination end-to-end using ES, which is consistent with our proposed method for this task. The combination attains a respectable mean score of 885, and as we can see in the third row of the above table, this approach also generalizes to a few of the unseen modifications of the CarRacing environment.

Our score on the base CarRacing task is lower than NetRand, but this is expected since our agent requires some amount of time steps to identify each of the inputs (which could be shuffled), while the NetRand and AttentionAgent agent will simply fail on the shuffled versions of CarRacing. Despite this, our method still compares favorably on the generalization performance.

To gain some insight into how the agent achieves its generalization ability, we visualize the attentions from the AttentionNeuron layer in the following figure:

<div style="text-align: left;">
<video class="b-lazy" data-src="assets/mp4/carracing_with_attention.mp4" type="video/mp4" autoplay muted playsinline loop style="width:100%;" ></video>
<video class="b-lazy" data-src="assets/mp4/fuji_attended_patch.mp4" type="video/mp4" autoplay muted playsinline loop style="width:100%;" ></video>
<figcaption style="text-align: left;">
<b>Attention visualization</b><br/>
We highlight the patches that receive the most attention.<br/>
Top: Attention plot in training environment.<br/>
Bottom: Attention plot in a test environment with unseen background.
</figcaption>
</div>

In CarRacing, the agent has learned to focus its attention (indicated by the highlighted patches) on the road boundaries which are intuitive to human beings and are critical to the task. Notice that the attended positions are consistent before and after the shuffling. This type of attention analysis can also be used to analyze failure cases too. More details about this visualization can be found in the Appendix.

______

## Related Work

Our work builds on ideas from various different areas:

**Self-organization** is a process where some form of global order emerges from local interactions between parts of an initially disordered system.
It is also a property observed in cellular automata (CA) <dt-cite key="neumann1966theory,codd2014cellular,conway1970game"></dt-cite>, which are mathematical systems consisting of a grid of cells that perform computation by having each cell communicate with its immediate neighbors and performing a local computation to update its internal state.
Such local interactions are useful in modeling complex systems <dt-cite key="wolfram1984cellular"></dt-cite> and have been applied to model non-linear dynamics in various fields <dt-cite key="chopard1998cellular"></dt-cite>. Cellular Neural Networks <dt-cite key="chua1988cellular"></dt-cite> were first introduced in the 1980s to use neural networks in place of the algorithmic cells in CA systems. They were applied to perform image processing operations with parallel computation. Eventually, the concept of self-organizing neural networks found its way into deep learning in the form of Graph Neural Networks (GNN) <dt-cite key="wu2020comprehensive,sanchezlengeling2021a"></dt-cite>.

Using modern deep learning tools, recent work demonstrates that *neural CA*, or self-organized neural networks performing only local computation, can generate (and re-generate) coherent images <dt-cite key="mordvintsev2020growing"></dt-cite> and voxel scenes <dt-cite key="zhang2021learning,sudhakaran2021growing"></dt-cite>, and even perform image classification <dt-cite key="randazzo2020selfclassifying"></dt-cite>. Self-organizing neural network agents have been proposed in the RL domain <dt-cite key="cheney2014unshackling,ohsawa2018neuron,ott2020giving,chang2020decentralized"></dt-cite>, with recent work demonstrating that shared local policies at the actuator level <dt-cite key="huang2020"></dt-cite>, through communicating with their immediate neighbors, can learn a global coherent policy for continuous control locomotion tasks.
While existing CA-based approaches present a modular, self-organized solution, they are *not* inherently permutation invariant. In our work, we build on neural CA, and enable each cell to communicate beyond its immediate neighbors via an attention mechanism that enables permutation invariance.

**Meta-learning** recurrent neural networks (RNN) <dt-cite key="hochreiter2001learning,haruno2001mosaic,duan2016rl,wang2016learning"></dt-cite> have been proposed to approach the problem of learning the learning rules for a neural network using the reward or error signal, enabling meta-learners to learn to solve problems presented outside of their original training domains. The goals are to enable agents to continually learn from their environments in a single lifetime episode, and to obtain much better data efficiency than conventional learning methods such as stochastic gradient descent (SGD). A meta-learned policy that can adapt the weights of a neural network to its inputs during inference time have been proposed in fast weights <dt-cite key="schmidhuber1992learning,schmidhuber1993self"></dt-cite>, associative weights <dt-cite key="ba2016using"></dt-cite>, hypernetworks <dt-cite key="ha2016hypernetworks"></dt-cite>, and Hebbian-learning <dt-cite key="miconi2018differentiable,miconi2020backpropamine"></dt-cite> approaches. Recently works <dt-cite key="sandler2021meta,kirsch2020meta"></dt-cite> combine ideas of self-organization with meta-learning RNNs, and have demonstrated that modular meta-learning RNN systems not only can learn to perform SGD-like learning rules, but can also discover more general learning rules that transfer to classification tasks on unseen datasets.

In contrast, the system presented here does not use an error or reward signal to meta-learn or fine-tune its policy. But rather, by using the shared modular building blocks from the meta-learning literature, we focus on learning or converting an existing policy to one that is permutation invariant, and we examine the characteristics such policies exhibit in a zero-shot setting, *without* additional training.

**Attention** can be viewed as an adaptive weight mechanism that alters the weight connections of a neural network layer based on what the inputs are. Linear *dot-product* attention has first been proposed for meta-learning <dt-cite key="schmidhuber1993reducing"></dt-cite>, and versions of linear attention with *softmax* nonlinearity appeared later <dt-cite key="graves2014neural,luong2015effective"></dt-cite>, now made popular with Transformer <dt-cite key="vaswani2017"></dt-cite>. The adaptive nature of attention provided the Transformer with a high degree of expressiveness, enabling it to learn inductive biases from large datasets and have been incorporated into state-of-the-art methods in natural language processing <dt-cite key="devlin2018bert,brown2020language"></dt-cite>, image recognition <dt-cite key="dosovitskiy2020image"></dt-cite> and generation <dt-cite key="esser2020taming"></dt-cite>, audio and video domains <dt-cite key="girdhar2019video,sun2019learning,jaegle2021perceiver"></dt-cite>.

Attention mechanisms have found many uses for RL <dt-cite key="sorokin2015deep,choi2017multi,zambaldi2018deep,mott2019towards,attentionagent2020"></dt-cite>. Our work here specifically uses attention to enable communication between arbitrary numbers of modules in an RL agent. While previous work <dt-cite key="velivckovic2017graph,monti2017geometric,zhang2018gaan,yun2019graph,joshi2020transformers,goyal2021recurrent"></dt-cite> utilized attention as a communication mechanism between independent neural network modules of a GNN, our work focuses on studying the permutation invariant properties of attention-based communication applied to RL agents. Related work <dt-cite key="liu2020pic"></dt-cite> used permutation invariant critics to enhance performance of multi-agent RL. Building on previous work on PI <dt-cite key="guttenberg2016permutation,zaheer2017deep"></dt-cite>, Set Transformers <dt-cite key="set2019"></dt-cite> investigated the use of attention explicitly for permutation invariant problems that deal with set-structured data, which have provided the theoretical foundation for our work.

______

## Discussion and Future Work

In this work, we investigate the properties of RL agents that can treat their observations as an arbitrarily ordered, variable-length list of sensory inputs. By processing each input stream independently, and consolidating the processed information using attention, our agents can still perform their tasks even if the ordering of the observations is randomly permuted several times during an episode, without explicitly training for frequent re-shuffling. We report results of performance versus shuffling frequency in the following table for each environment:

<div style="text-align: center;">
<img class="b-lazy" src=data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw== data-src="assets/png/table_shuffling_results.larger.png" style="display: block; margin: auto; width: 100%;"/>
<figcaption style="text-align: left;">
<b>Reshuffle observations during a roll-out</b><br/>
In each test episode, we reshuffle the observations every <i>t</i> steps.
For CartPole, we test for 1000 episodes because of its larger task variance. For the other tasks, we report mean and standard deviation from 100 tests.  All environments except for Atari Pong have a hard limit of 1000 time steps per episode. In Atari Pong, while the maximum length of an episode does not exist, we observed that an episode usually lasts for around 2500 steps.
</figcaption>
</div>

**Applications**&nbsp; By presenting the agent with shuffled, and even incomplete observations, we encourage it to interpret the meaning of each local sensory input and how they relate to the global context.
This could be useful in many real world applications. For example, such policies could avoid errors due to cross-wiring or complex, dynamic input-output mappings when being deployed in real robots. A similar setup to the CartPole experiment with extra noisy channels could enable a system that receives thousands of noisy input channels to identify the small subset of channels with relevant information.

**Limitations**&nbsp; For visual environments, patch size selection will affect both performance and computing complexity. We find that patches of 6x6 pixels work well for our tasks, as did 4x4 pixels to some extent, but single pixel observations fail to work. Small patch sizes also result in a large attention matrix which may be too costly to compute, unless approximations are used <dt-cite key="wang2020linformer,choromanski2020rethinking,xiong2021nystr"></dt-cite>.

Another limitation is that the permutation invariant property applies only to the inputs, and not to the outputs. While the ordering of the observations can be shuffled, the ordering of the actions cannot. For permutation invariant outputs to work, each action will require feedback from the environment, including reward information, in order to learn the relationship between itself and the environment.

**Societal Impact**&nbsp; Like most algorithms proposed in computer science and machine learning, our method can be applied in ways that will have potentially positive or negative impacts to society. While our small-scale, self-contained experiments study only the properties of RL agents that are permutation invariant to their observations, and we believe our results do not directly cause harm to society, the robustness and flexible properties of the method may be of use for data-collection systems that receive data from a large variable number of sensors. For instance, one could apply permutation invariant sensory systems to process data from millions of sensors for anomaly detection, which may lead to both positive or negative impacts, if used in applications such as large-scale sensor analysis for weather forecasting, or deployed in large-scale surveillance systems that could undermine our basic freedoms.

Our work also provides a way to view the Transformer <dt-cite key="vaswani2017"></dt-cite> through the lens of self-organizing neural networks. Transformers are known to have potentially negative societal impacts highlighted in studies about possible data-leakage and privacy vulnerabilities <dt-cite key="carlini2020extracting"></dt-cite>, malicious misuse and issues concerning bias and fairness <dt-cite key="bender2021dangers"></dt-cite>, and energy requirements for training these models <dt-cite key="strubell2019energy"></dt-cite>.

**Future Work**&nbsp; An interesting future direction is to also make the action layer have the same properties, and model each *motor neuron* as a module connected using attention. With such methods, it may be possible to train an agent with an arbitrary number of legs, or control robots with different morphology using a single policy that is also provided with a reward signal as feedback.
Moreover, our method accepts previous actions as a feedback signal in this work. However, the feedback signal is not restricted to the actions.
We look forward to seeing future works that include signals such as environmental rewards to train permutation invariant meta-learning agents that can adapt to not only changes in the observed environment, but also to changes to itself.

*If you would like to discuss any issues or give feedback, please visit the [GitHub](https://github.com/attentionneuron/attentionneuron.github.io/issues) repository of this page for more information.*
