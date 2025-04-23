import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid as trapz
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.special import ellipe
from scipy.optimize import root_scalar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import approx_fprime
from scipy.optimize import minimize




#==========================================================================

#                Defining constants and parameters

initial_radius = ((20/2) * 1e-03 + 1e-03/2)
theta = np.linspace(0, np.pi/2, 1000)
sin_theta = np.sin(theta)
desired_slope = 650
slopes = []
weight = 10
initial_delta = 0.008
learning_rate = 1e-12
epsilon = 0.00001              
num_iterations = 40
min_delta = 0.002           
max_delta = 0.02
min_2a = 0.02
max_2a = 0.04
min_radius = (20/2) * 1e-03 #+ 5e-03         
max_radius = (20/2) * 1e-03 + 5e-03


#==========================================================================
#                          Helper functions

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

def calculate_elliptic_integral(eccentricity, sin_theta, theta):
    return trapz(np.sqrt(1 - eccentricity**2 * sin_theta**2), theta)

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
    
def calculate_r2(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, y_pred)
    return r2



#==========================================================================

#                          Main Functions


def objetivo(variables, theta, sin_theta, curvature_loss_exp, 
                           TNB_Flat, error_function, desired_slope):
    totalLoss1 = []
    totalLoss2 = []
    two_a_value = []
    a1_value = []
    a2_value = []
    e2_value = []
    b1_value = []
    b2_value = []

    delta = variables[0]
    initial_radius = variables[1]
    length = 2 * np.pi * initial_radius


    for eccentricity in np.linspace(0, 0.999, 300):

        elliptic_integral = ellipe(eccentricity**2)
        a1, b1 = calculate_a_b(length, elliptic_integral, eccentricity)

        a2 = ((2 * b1) + delta) / 2
        two_a2 = 2 * a2
        #Choosing range for two_a2
        if (two_a2 < min_2a or two_a2 > max_2a):
            continue

        # Finding the eccentricity of the second ellipse 
        function_Ee = length / (4 * a2)
        
        if (function_Ee < 1 or function_Ee > np.pi / 2):
            continue

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
        return 999999

    mse, slope_difference, _, _ = fit_and_evaluate_losses(
        two_a_value, totalLoss_sum, desired_slope
    )

    result = mse * weight + slope_difference
    # print("mse: ",mse)
    # print("slope_dif: ", slope_difference)
    # print("delta: ", delta)
    # print ("result: ", result)
    # print( " ")

    return result





def optimize_gd(initial_delta, learning_rate, num_iterations, epsilon,
                theta, sin_theta, curvature_loss_exp, TNB_Flat,
                error_function, desired_slope, initial_radius):
    best_results = []

    
    # Initialize delta and initial_radius with physical constraints
    delta = np.clip(initial_delta, min_delta, max_delta)
    initial_radius = np.clip(initial_radius, min_radius, max_radius)

    best_delta = delta
    best_initial_radius = initial_radius
    variables = np.array([delta, initial_radius])

    # Define bounds for the variables
    bounds = [(min_delta, max_delta), (min_radius, max_radius)]
    
    res = minimize(objetivo, 
                variables, 
                args=(theta, sin_theta, curvature_loss_exp, TNB_Flat, error_function, desired_slope), 
                method='L-BFGS-B',
                bounds=bounds,
                options={'eps': epsilon})  

    # Update the variables with optimization results
    best_delta = res.x[0]
    best_initial_radius = res.x[1]
    current_metric = res.fun


    # Final evaluation with final delta and final initial radius
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

    best_length = 2 * np.pi * best_initial_radius

    for eccentricity in np.linspace(0, 0.95, 300):
        elliptic_integral = ellipe(eccentricity**2)
        a1, b1 = calculate_a_b(best_length, elliptic_integral, eccentricity)
        a2 = ((2 * b1) + best_delta) / 2
        two_a2 = 2 * a2

        if (two_a2 < min_2a or two_a2 > max_2a):
            continue

        function_Ee = best_length / (4 * a2)
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
        
    final_result = final_mse * weight + final_slope_diff
    return (best_delta, best_initial_radius, final_two_a_value, final_totalLoss_sum, 
            final_slope, final_intercept, final_a1_value, final_b1_value,
            final_a2_value, final_b2_value, final_e2_value, final_mse, final_slope_diff, final_result)

#==========================================================================
#
#                       Calculating the two ellipses

best_delta, best_initial_radius, final_two_a_value, final_totalLoss_sum, final_slope, final_intercept, a1, b1, a2, b2, e2, final_mse, final_slope_diff, final_result = optimize_gd (initial_delta, 
    learning_rate, num_iterations, epsilon, theta, sin_theta, curvature_loss_exp, TNB_Flat, error_function, desired_slope, initial_radius)
# #==========================================================================
# #
# #                               Results

# The best delta
print(" ")
print(f"The best delta is ", best_delta)
print("The best initial radius is ", best_initial_radius)
print("final mse: ", final_mse)
print("final slope: ", final_slope)
print("final result: ", final_result)

# Applying the metric
print(len(final_two_a_value), len(final_totalLoss_sum))
r2 = calculate_r2(final_two_a_value, final_totalLoss_sum)
print(f"R²: {r2}")

# Plot a1, a2, b1, b2 and e2
fig0, ax0 = plt.subplots(figsize=(13, 8))
ax0.plot(e2, a1, label="a1", marker="o")
ax0.plot(e2, a2, label="a2", marker="s")
ax0.plot(e2, b1, label="b1", marker="^")
ax0.plot(e2, b2, label="b2", marker="d")
ax0.set_xlabel("Eccentricity of the second ellipse")
ax0.set_ylabel("Values of a and b")
ax0.set_title("Behavior of a and b")
plt.grid(True, linestyle="--", alpha=0.6)

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
ax1.tick_params(axis='both', 
                which='major')
plt.grid(True, linestyle="--", alpha=0.6)

#Plot lines
y_desired = desired_slope * final_two_a_value + final_intercept
y_calculated = final_slope * final_two_a_value + final_intercept


fig2, ax2 = plt.subplots(figsize = (13,8))
ax2.plot(final_two_a_value, y_desired, color = "red", label = "desired")
ax2.plot(final_two_a_value, y_calculated, color = "blue", label = "calculated")
ax2.set_title('Fitted lines')
ax2.set_xlabel('2a [m]')
ax2.set_ylabel('Sum Loss [dB]')

plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
