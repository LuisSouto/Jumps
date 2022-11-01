# Option pricing with self-exciting jump processes

Option pricing of European and Bermudan put and call options using jump-diffusion models.

The notation and meaning of the parameters are extracted from Souto, Cirillo and
Oosterlee (2022): A new self-exciting jump-diffusion model for option pricing.
https://doi.org/10.48550/arXiv.2205.13321

The repository currently contains the following processes:
   - Heston diffusion (heston_class.py).
   - Q-Hawkes jump process (qhawkes_class.py).
   - Hawkes jump process (hawkes_class.py).
   - Poisson jump process (poisson_class.py).
   - Jump-diffusion process based on Heston diffusion and the previous jump models (jumpdiff_class.py)

The pricing methodology is based on the COS method (cos_method.py). Each class contains also a Monte Carlo (MC) style simulation function, so with few adaptations it is also possible to price using MC.

The main script is hestonjd_main.py, which compares the Bates (Heston+Poisson) model, the Heston-Hawkes model (HH) and the Heston-Queue-Hawkes (HQH). It mainly replicates the results of the paper mentioned above.
