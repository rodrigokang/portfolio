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

**Algorithm**: *Gradient Descent*

To approximate a local minimum of $\phi$ given an initial point
$\boldsymbol{\theta}_0$.

**INPUT** initial approximation $\boldsymbol{\theta}_0$; learning rate
$\eta$; tolerance $TOL$; maximum number of iterations $N$.

**OUTPUT** approximate minimizer $\boldsymbol{\theta}$ or a message of failure.

**Step 1** Set $k = 0$.

**Step 2** While $k < N$ do Steps 3–6.

**Step 3** Compute
$\mathbf{g} = \nabla\phi(\boldsymbol{\theta}_k)$.

**Step 4** If $\lVert\mathbf{g}\rVert < TOL$, then  
&nbsp;&nbsp;&nbsp;&nbsp;**OUTPUT** $(\boldsymbol{\theta}_k)$;  
&nbsp;&nbsp;&nbsp;&nbsp;**STOP**.

**Step 5** Set
$\boldsymbol{\theta}_{k+1}
= \boldsymbol{\theta}_k - \eta\mathbf{g}$.

**Step 6** Set $k = k + 1$.

**Step 7** **OUTPUT** ("Maximum number of iterations exceeded.");  
&nbsp;&nbsp;&nbsp;&nbsp;

**STOP**.