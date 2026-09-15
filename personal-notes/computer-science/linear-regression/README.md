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