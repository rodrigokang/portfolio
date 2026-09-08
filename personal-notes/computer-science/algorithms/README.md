# Gradient Descent  
***Technical Notes***  
**Author:** Rodrigo Kang

---

$$\theta_{t+1} =
\theta_t - \eta \nabla \phi(\theta_t)$$

```python
print("Hello world from Python!")
```

```r
print("Hello world from Python!")
```

```cpp
#include <iostream>

int main()
{
    std::cout << "Hello world from C++!\n";
    return 0;
}
```

---

**Algorithm:** *Gradient Descent*

Gradient descent is an iterative first-order optimization method that updates
the parameters in the direction of steepest decrease of the objective function.

**INPUT**

- initial approximation $\boldsymbol{\theta}_0$;
- learning rate $\eta$;
- tolerance $TOL$;
- maximum number of iterations $N$.

**OUTPUT**

- approximate minimizer $\boldsymbol{\theta}$ or a message of failure.

**Step 1** Set $k = 0$.

**Step 2** While $k < N$ do Steps 3–6.

**Step 3** Compute $\mathbf{g}_k = \nabla\phi(\boldsymbol{\theta}_k)$.

**Step 4** If $\lVert\mathbf{g}_k\rVert < TOL$, then  
&nbsp;&nbsp;&nbsp;&nbsp;**OUTPUT** $(\boldsymbol{\theta}_k)$;  
&nbsp;&nbsp;&nbsp;&nbsp;**STOP**.

**Step 5** Set
$\boldsymbol{\theta}_{k+1}
= \boldsymbol{\theta}_k - \eta\mathbf{g}_k$.

**Step 6** Set $k = k + 1$.

**Step 7** **OUTPUT** ("Maximum number of iterations exceeded.");  
&nbsp;&nbsp;&nbsp;&nbsp;**STOP**.

---