# Linear Regression  
***Technical Notes***  
**Author:** Rodrigo Kang

---

## Machine Learning Modelling

A central part of data science is finding patterns in data. To do that, we try to identify the underlying structure or relationships in the data and represent them through models. This process is called modelling. A model is an abstract and simplified representation of some aspect of reality.

Broadly speaking, there are two ways we can approach this: from first principles or from data:

$$\text{Modelling}
\longrightarrow
\begin{cases}
\text{First-Principles Modelling — Deductively} \\
\text{Data-Driven Modelling — Inductively}
\end{cases}$$

Machine learning belongs broadly to the data-driven approach. Depending on the information available and the learning problem, we can distinguish between supervised, unsupervised and reinforcement learning.

In supervised learning, we have labelled data, meaning that each observation includes a response or target variable. If the response is quantitative, we typically have a regression problem; if it is categorical, we have a classification problem.

In unsupervised learning, there is no response variable. Instead, we're interested in discovering structure in the data itself, for example through clustering or dimensionality reduction.

Reinforcement learning is different from both. An agent learns by interacting with an environment, taking actions and receiving rewards, with the goal of learning a policy that maximises cumulative reward over time.

$$\text{ML}
\longrightarrow
\begin{cases}
\text{SL}
&
\begin{cases}
\text{Regression}     & Y \text{ quantitative} \\
\text{Classification} & Y \text{ categorical}
\end{cases}
\\
\text{UL} & \text{no response } Y
\\
\text{RL} & (s_t,a_t,r_t)
\end{cases}$$

In practice, these approaches don't have to be mutually exclusive. In one of the projects I've worked on, for example, we were interested in predicting microbiologically influenced corrosion, or MIC. We combined mechanistic models, such as Lotka–Volterra and Monod models, with data-driven regression and classification approaches. The same general idea appears in areas such as weather forecasting, where physical models of atmospheric dynamics can be combined with machine-learning methods.

## Notation

Throughout these notes, $N$ denotes the number of observations and $M$ the number of predictors.

The predictor variables are denoted by

$$X = \left(X_1,X_2,\dots,X_M\right)$$

For the $i$-th observation, the predictor values are collected in the vector

$$
\mathbf{x}_i
=
\begin{pmatrix}
x_{i1} \\
x_{i2} \\
\vdots \\
x_{iM}
\end{pmatrix}
=
\left(
x_{i1},
x_{i2},
\dots,
x_{iM}
\right)^T
\in\mathbb{R}^M,
$$

where $x_{ij}$ denotes the observed value of predictor $j$ for observation $i$.

The complete set of predictors can then be arranged into the $N\times M$ matrix

$$\mathbf{X}
=
\begin{pmatrix}
\mathbf{x}_1^T \\
\mathbf{x}_2^T \\
\vdots \\
\mathbf{x}_N^T
\end{pmatrix}
=
\begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1M} \\
x_{21} & x_{22} & \cdots & x_{2M} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N1} & x_{N2} & \cdots & x_{NM}
\end{pmatrix}
\in\mathbb{R}^{N\times M}.$$

Each row of $\mathbf{X}$ represents one observation, while each column represents one predictor. In particular, $\mathbf{x}_i^T$ is the $i$-th row of the predictor matrix.

For a quantitative response, the response variable is denoted by

$$Y\in\mathbb{R}.$$

The observed response for observation $i$ is denoted by $y_i$, and the complete set of responses is collected in the vector

$$\mathbf{y}
=
\begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{pmatrix}
\in\mathbb{R}^N.$$

For a categorical response, the response variable is denoted by

$$C\in\mathcal{C},$$

where $\mathcal{C}$ is the set of possible classes. For example, for a problem with $K$ classes,

$$\mathcal{C}
=
\{1,2,\dots,K\}.$$

The observed class for observation $i$ is denoted by $c_i\in\mathcal{C}$, and the complete set of observed class labels is collected in

$$\mathbf{c}
=
\begin{pmatrix}
c_1 \\
c_2 \\
\vdots \\
c_N
\end{pmatrix}
\in\mathcal{C}^N.$$

Given a new predictor vector

$$\mathbf{x}\in\mathbb{R}^M,$$

a regression model produces an estimate of the quantitative response,

$$\hat{Y}
=
\hat{\varphi}(\mathbf{x}),$$

where

$$\hat{\varphi}:\mathbb{R}^M\rightarrow\mathbb{R}.$$

Similarly, a classification model produces an estimate of the categorical response,

$$\hat{C}
=
\hat{\varphi}(\mathbf{x}),$$

where

$$\hat{\varphi}:\mathbb{R}^M\rightarrow\mathcal{C}.$$

## Goal of Supervised Learning

In supervised learning, each observation is paired with its corresponding response. For a regression problem, the dataset is denoted by

$$\mathcal{D}
=
\left\{
(\mathbf{x}_i,y_i)
\right\}_{i=1}^{N},$$

where $\mathbf{x}_i\in\mathbb{R}^M$ is the predictor vector for observation $i$ and $y_i\in\mathbb{R}$ is its corresponding quantitative response.

For a classification problem, the dataset is denoted by

$$\mathcal{D}
=
\left\{
(\mathbf{x}_i,c_i)
\right\}_{i=1}^{N},$$

where $c_i\in\mathcal{C}$ is the class label associated with observation $i$.

When the dataset is partitioned for model development and evaluation, the corresponding subsets are denoted by

$$\mathcal{D}_{\mathrm{train}},
\qquad
\mathcal{D}_{\mathrm{val}},
\qquad
\mathcal{D}_{\mathrm{test}},$$

for the training, validation, and test sets, respectively.

The dataset can also be represented in tabular form:

<div align="center">

| Observation | $X_1$       | $X_2$    | $\cdots$ | $X_M$     | $Y$      |
|:-----------:|:-----------:|:--------:|:--------:|:---------:|:--------:|
| $1$         | $x_{11}$    | $x_{12}$ | $\cdots$ | $x_{1M}$  | $y_1$    |
| $2$         | $x_{21}$    | $x_{22}$ | $\cdots$ | $x_{2M}$  | $y_2$    |
| $\vdots$    | $\vdots$    | $\vdots$ | $\vdots$ | $\vdots$  | $\vdots$ |
| $N$         | $x_{N1}$    | $x_{N2}$ | $\cdots$ | $x_{NM}$  | $y_N$    |

</div>

The columns $X_1,X_2,\dots,X_M$ correspond to the predictors, while $Y$ corresponds to the response variable. Each row represents one observation, so that row $i$ corresponds to the pair

$$(\mathbf{x}_i,y_i),$$

where

$$\mathbf{x}_i
=
(x_{i1},x_{i2},\dots,x_{iM})^T.$$

For a classification problem, the same structure applies, replacing the quantitative response $Y$ and its observed values $y_i$ with the categorical response $C$ and class labels $c_i$, respectively.

For a regression problem, we can represent the relationship between the response and the predictors as

$$Y
=
\varphi(X)
+
\varepsilon,$$

where $\varphi$ represents the systematic relationship between the predictors and the response, and $\varepsilon$ represents the random component that cannot be explained by the predictors.

For the $i$-th observation,

$$y_i
=
\varphi(\mathbf{x}_i)
+
\varepsilon_i.$$

The goal is to use the available data to estimate the unknown relationship $\varphi$. The resulting estimate is denoted by

$$\hat{\varphi}.$$

Given a new predictor vector $\mathbf{x}$, the corresponding prediction is then

$$\hat{Y}
=
\hat{\varphi}(\mathbf{x}).$$

For classification, the same general goal applies: we want to learn the relationship between the predictors and the categorical response, although the relationship is not generally expressed through the additive form above.

### Estimating the Relationship

There are different ways of approaching the estimation of $\varphi$. At a high level, we can distinguish between prespecifying a functional form and allowing a more flexible relationship to be learned from the data:

$$\text{Estimating } \varphi
\longrightarrow
\begin{cases}
\text{Prespecifying a Functional Form} \\
\text{Learning a Flexible Relationship from Data}.
\end{cases}$$

In the first approach, the general structure of the relationship is specified before fitting the model. For example, we may assume a linear relationship,

$$\varphi_{\boldsymbol{\theta}}(\mathbf{x})
=
\theta_0
+
\sum_{j=1}^{M}\theta_jx_j.$$

The learning problem then reduces to estimating the parameters

$$\boldsymbol{\theta}
=
(\theta_0,\theta_1,\dots,\theta_M)^T.$$

Linear regression is a canonical example of this approach. Polynomial regression and other explicitly parameterised functional forms follow the same general idea.

Alternatively, we may use a more flexible model class and allow the data to determine a larger part of the structure of the relationship. Examples include k-nearest neighbours, decision trees, random forests, boosting methods, and neural networks.

These two approaches should not be interpreted as assumptions versus no assumptions. Every learning method imposes some restrictions or preferences on the relationships that can be learned. This is known as **inductive bias**.

For example, linear regression restricts the model to linear relationships, k-nearest neighbours relies on the idea that nearby observations in predictor space should tend to have similar responses, decision trees represent relationships through recursive partitions of the predictor space, and neural networks restrict the learned relationship through their architecture and composition of transformations.

Once a model has been specified, we need a criterion for determining how well it fits the observed data. This is usually expressed through a cost function, also commonly referred to as a loss function or objective function, which we denote by $J$.

The general strategy is to find the model that minimises this function:

$$\hat{\varphi}
=
\underset{\varphi}{\arg\min}
\;
J(Y, \varphi(X)).$$

When the model is parameterised by a vector of parameters $\boldsymbol{\theta}$, this can instead be written as

$$\hat{\boldsymbol{\theta}}
=
\underset{\boldsymbol{\theta}}{\arg\min}
\;
J(\boldsymbol{\theta}).$$

The particular form of $J$ depends on the model and the problem. For example, squared error is commonly used for regression, while cross-entropy is commonly used for classification.