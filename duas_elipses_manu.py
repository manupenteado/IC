import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid as trapz
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.special import ellipe
from scipy.optimize import root_scalar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



#==========================================================================

#                Defining constants and parameters

#deltas = np.linspace(0.00001, 0.021, 25)
initial_radius = (20/2) * 1e-03
length = 2 * np.pi * initial_radius
theta = np.linspace(0, np.pi/2, 1000)
sin_theta = np.sin(theta)
desired_slope = 300
slopes = []
weight = 10
initial_delta = 0.008
learning_rate = 0.000000001        
epsilon = 0.00001              
num_iterations = 100        
min_delta = 0.002           
max_delta = 0.02      

#==========================================================================

#                          Helper Functions

def TNB_Flat(x, y, t):
    # dr is the differential of the curve
    dr = np.matrix([np.gradient(x), np.gradient(y)])
    ds = np.power(np.sum(np.square(dr), 0), 0.5)  # Arc length associated with each point. ||dr||.
    T = np.divide(dr, ds)  # Unit tangent vector. e aponta na direção da curva

    # s is the arc of the length, it grows with T
    T0 = np.zeros(len(x))
    T1 = np.zeros(len(x))
    
    # Separating in x and y
    for i in range(0, len(x), 1):
        T0[i] = T[0, i]
        T1[i] = T[1, i]

    # dT is the differential of the tangent vector
    dT = np.c_[np.gradient(T0), np.gradient(T1)]  # T'(t).
    dT = np.transpose(dT)
    dTds = np.divide(dT, ds)

    # Calculate first derivatives
    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)

    # Calculate second derivatives
    d2x_dt2 = np.gradient(dx_dt, t)
    d2y_dt2 = np.gradient(dy_dt, t)

    # Form the first and second derivative vectors
    drdt = np.vstack((dx_dt, dy_dt)).T
    d2rdt2 = np.vstack((d2x_dt2, d2y_dt2)).T

    # Expand the 2D vectors to 3D by adding a third dimension of zeros
    drdt_3d = np.hstack((drdt, np.zeros((drdt.shape[0], 1))))
    d2rdt2_3d = np.hstack((d2rdt2, np.zeros((d2rdt2.shape[0], 1))))

    # Compute the cross product in 3D
    cross_product = np.cross(drdt_3d, d2rdt2_3d)

    # Ensure cross_product is a 2D array
    if cross_product.ndim == 1:
        cross_product = cross_product.reshape(-1, 1)

    # Compute the numerator of the curvature formula (norm of the cross product)
    numerator = np.linalg.norm(cross_product, axis=1)

    # Compute the denominator of the curvature formula (norm of the first derivative to the power of 3)
    denominator = np.linalg.norm(drdt, axis=1) ** 3

    # Compute the curvature (t)
    kappa_t = numerator / denominator
    kappa = np.power(sum(np.square(dTds), 0), 0.5)

    # Normal vector
    N = np.divide(dTds, kappa)  # Unit normal vector.

    return T, N, kappa, kappa_t, ds

# Method to define ellipse
def ellipse(a, b, t):
    x = a * np.cos(t + np.pi/2)
    y = b * np.sin(t + np.pi/2)
    return x, y

def curvature_loss_exp(x):
    a1 = 4938.863079635098
    b1 = -660.2338027131779
    a2 = -4.724968706275023
    mu = 0.008342737443501743
    sigma = 0.0018582333914787604
 
    x = 1 / x
 
    x = np.asarray(x)
    mu = np.asarray(mu)
 
    return a1 * np.exp(b1 * x) + a2 * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
 

# def curvature_loss_exp(curvature):
#     a1 = 1525.965951223269
#     b1 = -869.0646471499498
#     a2 = 1525.9295489081296
#     b2 = -869.0650046536391

#     return a1 * np.exp(b1 * (1 / curvature)) + a2 * np.exp(b2 * (1 / curvature))

# Function to compute the elliptic integral
def calculate_elliptic_integral(eccentricity, sin_theta, theta):
    return trapz(np.sqrt(1 - eccentricity**2 * sin_theta**2), theta)

# Calculating a and b of an ellipse
def calculate_a_b(length, elliptic_integral, eccentricity):
    b = (1 / 4) * length * np.sqrt(1 - eccentricity**2) / elliptic_integral
    a = b / np.sqrt(1 - eccentricity**2)
    return a, b

def error_function(e, E):
    return ellipe(e**2) - E

def calculate_eccentricity(error_function, function_Ee):
    # e2 inicial [a, b] onde a função muda de sinal
    a = 0
    b = 1

    result = root_scalar(
        lambda e: error_function(e, function_Ee),
        method='brentq',
        bracket=[a, b],
        xtol=1e-7
    )

    eccentricity = result.root
    return eccentricity

# Calculate the curvature and its losses
def calculate_curvature_and_loss(x, y, TNB_Flat, curvature_loss_exp):
    T, N, kappa_TNB, _, ds = TNB_Flat(x, y, np.linspace(0, 2 * np.pi, 500))
    curvature_loss = curvature_loss_exp(kappa_TNB)
    s = cumtrapz(ds, initial=0)
    total_loss = trapz(curvature_loss, s)
    return kappa_TNB, total_loss, s

# Used to calculate the Mean Squared Error between the y value (sum of losses) and a straight line
def mse_loss(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)  
    y_pred = model.predict(x.reshape(-1, 1))  
    mse = np.mean((y - y_pred) ** 2)  
    return mse


def fit_and_evaluate_losses(two_a_value, totalLoss_sum, desired_slope):
    
    x = two_a_value.reshape(-1, 1)
    y = totalLoss_sum
    mse = mse_loss(x, y)
    
    model = LinearRegression()
    model.fit(x, y)
    calculated_slope = model.coef_[0]
    calculated_intercept = model.intercept_
    slope_difference = abs(calculated_slope - desired_slope)

    return mse, slope_difference, calculated_slope, calculated_intercept
    

# Linearity metric: R-squared (R²) / Coefficient of Determination
def calculate_r2(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_pred)
    return r2



#==========================================================================

#                          Main Functions
def compute_combined_metric(delta, length, theta, sin_theta, curvature_loss_exp, 
                           TNB_Flat, error_function, desired_slope):
    totalLoss1 = []
    totalLoss2 = []
    two_a_value = []
    a1_value = []
    a2_value = []
    e2_value = []
    b1_value = []
    b2_value = []    

    for eccentricity in np.linspace(0, 0.999, 300):
        elliptic_integral = ellipe(eccentricity**2)
        a1, b1 = calculate_a_b(length, elliptic_integral, eccentricity)

        a2 = ((2 * b1) + delta) / 2
        two_a2 = 2 * a2
        
        # Choosing range for two_a2
        # if (two_a2 < 0.022 or two_a2 > 0.024):
        #     continue

        # Finding the eccentricity of the second ellipse 
        function_Ee = length / (4 * a2)
        
        if (function_Ee < 1 or function_Ee > np.pi / 2):
            break

        e2 = calculate_eccentricity(error_function, function_Ee)

        b2 = a2 * (np.sqrt(1 - (e2 ** 2)))

        # Generate ellipses
        t = np.linspace(0, 2 * np.pi, 500)
        x1, y1 = ellipse(a1, b1, t)
        x2, y2 = ellipse(a2, b2, t)

        kappa1, loss1, _ = calculate_curvature_and_loss(x1, y1, TNB_Flat, curvature_loss_exp)
        kappa2, loss2, _ = calculate_curvature_and_loss(x2, y2, TNB_Flat, curvature_loss_exp)

        # Verify the value of the radius
        if any([(1/np.max(kappa1)) < 0.00237, (1/np.max(kappa2)) < 0.00237]):
            continue 

        a1_value.append(a1)
        a2_value.append(a2)
        b1_value.append(b1)
        b2_value.append(b2)
        totalLoss1.append(loss1)
        totalLoss2.append(loss2)
        two_a_value.append(two_a2)
        e2_value.append(e2)

        
    totalLoss_sum = np.array(totalLoss1) + np.array(totalLoss2)
    two_a_value = np.array(two_a_value)

    if len(two_a_value) < 10 or len(totalLoss_sum) < 10:
        print("Fail")
        print("delta: ", delta)
        return float('inf')

    mse, slope_difference, _, _ = fit_and_evaluate_losses(
        two_a_value, totalLoss_sum, desired_slope
    )

    print("mse: ",mse)
    print("slope_dif: ", slope_difference)
    print("delta: ", delta)

    result = mse * weight + slope_difference
    print (result)

    return result








def optimize_delta_gd(initial_delta, learning_rate, num_iterations, epsilon,
                     length, theta, sin_theta, curvature_loss_exp, TNB_Flat,
                     error_function, desired_slope):
    
    """
    Gradient descent optimization for delta parameter
    
    Optimization Strategy:
    1. Uses finite difference gradient estimation
    2. Maintains delta within physical bounds [min_delta, max_delta]
    3. Tracks best performing delta through iterations
    4. Final evaluation with best delta
    
    Parameters:
    learning_rate : float - Step size for gradient updates
    epsilon : float - Perturbation size for gradient calculation
    num_iterations : int - Maximum optimization steps
    """
    
    # Initialize delta with physical constraints
    delta = np.clip(initial_delta, min_delta, max_delta)
    best_delta = delta
    best_metric = float('inf')
    history = []
    i = 0
    for _ in range(num_iterations):

        # Bounded parameter perturbations
        delta_plus = delta + epsilon
        delta_minus = delta - epsilon

        # Calculate finite difference gradient
        metric_plus = compute_combined_metric(
            delta_plus, 
            length, theta, sin_theta, curvature_loss_exp,
            TNB_Flat, error_function, desired_slope
        )
        metric_minus = compute_combined_metric(
            delta_minus,
            length, theta, sin_theta, curvature_loss_exp,
            TNB_Flat, error_function, desired_slope
        )
        print("oi")

        # Handle numerical instability
        if np.isinf(metric_plus) or np.isinf(metric_minus):
            #print(i)
            i = i + 1
            continue

        # Central difference gradient estimation
        gradient = (metric_plus - metric_minus) / (2 * epsilon)
        # Update delta with gradient descent
        delta = np.clip(delta - learning_rate * gradient, min_delta, max_delta)
        print("novo delta: ", delta)
        current_metric = compute_combined_metric(
            delta,
            length, theta, sin_theta, curvature_loss_exp,
            TNB_Flat, error_function, desired_slope
        )
        
        if current_metric < best_metric:
            best_metric = current_metric
            best_delta = delta
            print("Best delta: ", best_delta)
        
        history.append(current_metric)

    # Final evaluation with final delta
    final_totalLoss1 = []
    final_totalLoss2 = []
    final_two_a_value = []
    final_a1_value = []
    final_a2_value = []
    final_e2_value = []
    final_b1_value = []
    final_b2_value = []
    final_slope = None
    final_intercept = None

    for eccentricity in np.linspace(0, 0.95, 300):
        elliptic_integral = ellipe(eccentricity**2)
        a1, b1 = calculate_a_b(length, elliptic_integral, eccentricity)
        a2 = ((2 * b1) + best_delta) / 2
        two_a2 = 2 * a2

        # if (two_a2 < 0.022 or two_a2 > 0.024):
        #     # print("Fail")
        #     continue

        function_Ee = length / (4 * a2)
        if (function_Ee < 1 or function_Ee > np.pi / 2):
            continue

        e2 = calculate_eccentricity(error_function, function_Ee)
        b2 = a2 * (np.sqrt(1 - (e2 ** 2)))

        # Generate ellipses and calculate curvature/loss
        t = np.linspace(0, 2 * np.pi, 500)
        x1, y1 = ellipse(a1, b1, t)
        x2, y2 = ellipse(a2, b2, t)

        kappa1, loss1, _ = calculate_curvature_and_loss(x1, y1, TNB_Flat, curvature_loss_exp)
        kappa2, loss2, _ = calculate_curvature_and_loss(x2, y2, TNB_Flat, curvature_loss_exp)

        if any([(1/np.max(kappa1)) < 0.00237, (1/np.max(kappa2)) < 0.00237]):
            continue

        # Store values in final arrays
        final_a1_value.append(a1)
        final_a2_value.append(a2)
        final_b1_value.append(b1)
        final_b2_value.append(b2)
        final_totalLoss1.append(loss1)
        final_totalLoss2.append(loss2)
        final_two_a_value.append(two_a2)
        final_e2_value.append(e2)

    # Calculate final metrics
    final_totalLoss_sum = np.array(final_totalLoss1) + np.array(final_totalLoss2)
    final_two_a_value = np.array(final_two_a_value)
    
    final_mse, final_slope_diff, final_slope, final_intercept = fit_and_evaluate_losses(
            final_two_a_value, final_totalLoss_sum, desired_slope
        )
        

    return (best_delta, final_two_a_value, final_totalLoss_sum, 
            final_slope, final_intercept, final_a1_value, final_b1_value,
            final_a2_value, final_b2_value, final_e2_value)
#==========================================================================
#
#                       Calculating the two ellipses

best_delta, final_two_a_value, final_totalLoss_sum, final_slope, final_intercept, a1, b1, a2, b2, e2 = optimize_delta_gd (initial_delta, learning_rate, num_iterations, epsilon, length, theta, sin_theta, curvature_loss_exp, TNB_Flat, error_function, desired_slope)
# #==========================================================================
# #
# #                               Results

# The best delta
print(f"The best delta is {best_delta}")

# Applying the metric
print(len(final_two_a_value), len(final_totalLoss_sum))
r2 = calculate_r2(final_two_a_value, final_totalLoss_sum)
print(f"R²: {r2}")

# Plot a1, a2, b1, b2 and e2
plt.figure(figsize=(10, 6))
plt.plot(e2, a1, label="a1", marker="o")
plt.plot(e2, a2, label="a2", marker="s")
plt.plot(e2, b1, label="b1", marker="^")
plt.plot(e2, b2, label="b2", marker="d")
plt.xlabel("Eccentricity of the second ellipse")
plt.ylabel("Values of a and b")
plt.title("Behavior of a and b")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Plot 2a vs sum of total losses
fig1, ax1 = plt.subplots(figsize=(13, 8))
ax1.plot(
    final_two_a_value, 
    final_totalLoss_sum, 
    label=f'Sum of Curvature Losses [dB] (optimized delta = {best_delta})', 
    color='blue',
    linewidth=2)
ax1.set_title('Sum of Curvature Losses vs 2a (Optimized)')
ax1.set_xlabel('2a [m]')
ax1.set_ylabel('Sum Loss [dB]')
ax1.grid(True)
plt.tight_layout()
ax1.tick_params(axis='both', 
                which='major')
plt.legend()
plt.show()


"""o limite do delta sempre vai ser o melhor delta. por que?
quanto mais voce aumenta o limite do delta, menos amostras vc tera. quanto menos amostras, mais perto de uma reta vai estar...
por exemplo, se tiver apenas duas amostras, a distancia entre elas vai ser sempre uma reta, enão vai continuar sendo assim ate que a elipse pare de existir.
como resolver? talvez colocando um limite minimo de amostras permitidos para analisar quão bem se encaixa numa reta"""

"""e se eu pegasse o que tem mais samples e r2??? 
ex: The best delta is 0.015
280 280
R²: 0.6081005728350152
The best delta is 0.017
226 226
R²: 0.5664841764413471

por que eles escolheria o 0.017 ao inves do 15 se o 15, com mais amostras, se mostrou um fit melhor???
"""

"""uma vez retornando infinito, acabou. o delta nunca vai conseguir ser atualizado, pq vai sempre ficar voltando pro mesmo, por isso so
roda as duas primeiras vezes

sera que teria um jeito de implementar esse tipo de algoritmo, ja que, variando um dado, as vezes a informação referente a ele
simplesmente n existe? ele precisa da informação anterior pra andar e, se ela nao existe, como faz??"""