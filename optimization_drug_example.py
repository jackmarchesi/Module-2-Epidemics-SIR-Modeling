# drug efficacy optimization example for BME 2315
# made by Lavie, fall 2025

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt


#%% define drug models

# define toxicity levels for each drug (lambda)
metformin_lambda = 0.5

lisinopril_lambda = 0.8

<<<<<<< HEAD
escitalopram_lambda = 0.9
=======
escitalopram_lambda = 0.3
>>>>>>> 610da3ad3c1e112a0f529233326b4450c548ea55

def metformin(x):   # mild toxicity, moderate efficacy
    efficacy = 0.8 * np.exp(-0.1*(x-5)**2)
    toxicity = 0.2 * x**2 / 100
    return efficacy - metformin_lambda * toxicity
def lisinopril(x):  # strong efficacy, higher toxicity
    efficacy = np.exp(-0.1*(x-7)**2)
    toxicity = 0.3 * x**2 / 80
    return efficacy - lisinopril_lambda * toxicity
def escitalopram(x):  # weaker efficacy, low toxicity
    efficacy = 0.6 * np.exp(-0.1*(x-4)**2)
    toxicity = 0.1 * x**2 / 120
    return efficacy - escitalopram_lambda * toxicity
def drug_effects(x):
    return metformin(x) + lisinopril(x) + escitalopram(x)

#%% plot drug efficacies
x = np.linspace(0, 15, 100)
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(x, metformin(x), label='Metformin', color='blue')
plt.plot(x, lisinopril(x), label='Lisinopril', color='orange')
plt.plot(x, escitalopram(x), label='Escitalopram', color='green')
plt.plot(x, drug_effects(x), label='Combined Effect', color='red', linestyle='--')
plt.title('Drug Efficacy vs Dosage')
plt.xlabel('Dosage (mg)')
plt.ylabel('Net Effect')
plt.legend()

# %% Find optimal dosages for each drug

# First method: Steepest Ascent using the update rule

<<<<<<< HEAD
=======

>>>>>>> 610da3ad3c1e112a0f529233326b4450c548ea55
# first, need the first derivative (gradient)
def gradient(f, x, h=1e-4):
    """Central difference approximation for f'(x)."""
    return (f(x + h) - f(x - h)) / (2*h)

def steepest_ascent(f, x0, h_step=0.1, tol=1e-6, max_iter=1000):
    x = x0 # update initial guess
    for i in range(max_iter):
        grad = gradient(f, x)
        x_new = x + h_step * grad     
        
        if abs(x_new - x) < tol:      # convergence condition, when solution is 0
            print(f"Converged in {i+1} iterations.")
            break
            
        x = x_new
    return x, f(x)

# metformin
opt_dose_metformin, opt_effect_metformin = steepest_ascent(metformin, x0=1.0)
print(f"Steepest Ascent Method - Optimal Metformin Dose: {opt_dose_metformin:.2f} mg")
print(f"Steepest Ascent Method - Optimal Metformin Effect: {opt_effect_metformin*100:.2f}%")

# lisinopril
opt_dose_lisinopril, opt_effect_lisinopril = steepest_ascent(lisinopril, x0=1.0)
print(f"Steepest Ascent Method - Optimal Lisinopril Dose: {opt_dose_lisinopril:.2f} mg")
print(f"Steepest Ascent Method - Optimal Lisinopril Effect: {opt_effect_lisinopril*100:.2f}%")

# escitalopram
opt_dose_escitalopram, opt_effect_escitalopram = steepest_ascent(escitalopram, x0=1.0)
print(f"Steepest Ascent Method - Optimal Escitalopram Dose: {opt_dose_escitalopram:.2f} mg")
print(f"Steepest Ascent Method - Optimal Escitalopram Effect: {opt_effect_escitalopram*100:.2f}%")

# %% Newton's method

# requires second derivative
def second_derivative(f, x, h=1e-4):
    """Central difference approximation for f''(x)."""
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

def newtons_method(f, x0, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        grad = gradient(f, x)
        hess = second_derivative(f, x)
        
        if hess == 0:  # avoid division by zero
            print("Zero second derivative. No solution found.")
            return x, f(x)
        
        x_new = x - grad / hess
        
        if abs(x_new - x) < tol:
            print(f"Converged in {i+1} iterations.")
            break
            
        x = x_new
    return x, f(x)

# metformin
opt_dose_metformin_nm, opt_effect_metformin_nm = newtons_method(metformin, x0=1.0)
print(f"Newton's Method - Optimal Metformin Dose: {opt_dose_metformin_nm:.2f} mg")
print(f"Newton's Method - Optimal Metformin Effect: {opt_effect_metformin_nm*100:.2f}%")                

# lisinopril
opt_dose_lisinopril_nm, opt_effect_lisinopril_nm = newtons_method(lisinopril, x0=1.0)
print(f"Newton's Method - Optimal Lisinopril Dose: {opt_dose_lisinopril_nm:.2f} mg")
print(f"Newton's Method - Optimal Lisinopril Effect: {opt_effect_lisinopril_nm*100:.2f}%")

# escitalopram
opt_dose_escitalopram_nm, opt_effect_escitalopram_nm = newtons_method(escitalopram, x0=1.0)
print(f"Newton's Method - Optimal Escitalopram Dose: {opt_dose_escitalopram_nm:.2f} mg")
print(f"Newton's Method - Optimal Escitalopram Effect: {opt_effect_escitalopram_nm*100:.2f}%")
<<<<<<< HEAD
=======

#Task 2: Compare the two methods on the combined drug effect optimization problem

#The following code block compares the two methods on the combined drug effect, which is a more complex optimization problem due to the interaction of multiple drugs.
print("\n--- Combined Drug Effect Optimization ---")

# Steepest Ascent on combined effect
print("\nSteepest Ascent:")
opt_dose_combined_sa, opt_effect_combined_sa = steepest_ascent(drug_effects, x0=1.0)
print(f"Steepest Ascent - Optimal Combined Dose: {opt_dose_combined_sa:.2f} mg")
print(f"Steepest Ascent - Optimal Combined Effect: {opt_effect_combined_sa*100:.2f}%")

# Newton's Method on combined effect
print("\nNewton's Method:")
opt_dose_combined_nm, opt_effect_combined_nm = newtons_method(drug_effects, x0=1.0)
print(f"Newton's Method - Optimal Combined Dose: {opt_dose_combined_nm:.2f} mg")
print(f"Newton's Method - Optimal Combined Effect: {opt_effect_combined_nm*100:.2f}%")

# Observe: Newton's method converges in far fewer iterations than steepest ascent
# because it uses curvature (second derivative) information to take smarter steps.

#%% Effect of max_iter on result accuracy

print("\n--- Effect of max_iter on Steepest Ascent ---")
for n_iter in [5, 10, 50, 1000]:
    dose, effect = steepest_ascent(metformin, x0=1.0, max_iter=n_iter)
    print(f"max_iter={n_iter:4d} -> dose: {dose:.4f} mg, effect: {effect*100:.4f}%")


# TASK 3: Lambda optimization loop — tune one drug to match combined optimal dose

#%% Find the best lambda for Metformin to match the combined effect's optimal dose

target_dose = opt_dose_combined_nm  # use Newton's result as the target
print(f"\n--- Lambda Tuning for Metformin ---")
print(f"Target (combined) optimal dose: {target_dose:.4f} mg\n")

lambda_values = np.linspace(0.1, 2.0, 100)  # sweep lambda from 0.1 to 2.0
best_lambda = None
best_dose_diff = np.inf
best_dose_achieved = None

for lam in lambda_values:
    # redefine metformin with the candidate lambda
    def metformin_test(x):
        efficacy = 0.8 * np.exp(-0.1*(x-5)**2)
        toxicity = 0.2 * x**2 / 100
        return efficacy - lam * toxicity

    opt_dose, _ = newtons_method(metformin_test, x0=1.0)
    diff = abs(opt_dose - target_dose)

    if diff < best_dose_diff:
        best_dose_diff = diff
        best_lambda = lam
        best_dose_achieved = opt_dose

print(f"Best lambda for Metformin: {best_lambda:.4f}")
print(f"Achieved optimal dose:     {best_dose_achieved:.4f} mg")
print(f"Target dose:               {target_dose:.4f} mg")
print(f"Difference:                {best_dose_diff:.4f} mg")

#%% Plot tuned Metformin vs combined effect

def metformin_best(x):
    efficacy = 0.8 * np.exp(-0.1*(x-5)**2)
    toxicity = 0.2 * x**2 / 100
    return efficacy - best_lambda * toxicity

fig2, ax2 = plt.subplots(figsize=(10, 6))
plt.plot(x, drug_effects(x), label='Combined Effect', color='red', linestyle='--') # plot combined effect for reference
plt.plot(x, metformin(x), label=f'Metformin (original λ={metformin_lambda})', color='blue', alpha=0.4) # original metformin for comparison
plt.plot(x, metformin_best(x), label=f'Metformin (tuned λ={best_lambda:.3f})', color='blue') # tuned metformin with best lambda
plt.axvline(target_dose, color='gray', linestyle=':', label=f'Target dose: {target_dose:.2f} mg') # target dose from combined effect optimization
plt.axvline(best_dose_achieved, color='blue', linestyle=':', alpha=0.6,
            label=f'Tuned dose: {best_dose_achieved:.2f} mg') # achieved dose from tuning lambda
plt.title('Metformin Lambda Tuning to Match Combined Optimal Dose') # title for the plot, followed by x,y labels and legend
plt.xlabel('Dosage (mg)')
plt.ylabel('Net Effect')
plt.legend()
plt.show()
# This code block demonstrates how adjusting the toxicity parameter (lambda) for Metformin can shift its optimal dose to better align with the optimal dose of the combined drug effect. By tuning lambda, we can effectively "steer" the optimization landscape to achieve a desired outcome.
# If the best lambda is significantly different from the original, it suggests that the original toxicity assumption for Metformin may not be optimal for achieving the best combined effect.
>>>>>>> 610da3ad3c1e112a0f529233326b4450c548ea55
