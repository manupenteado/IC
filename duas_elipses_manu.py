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

deltas = np.linspace(0.00001, 0.021, 25)
initial_radius = (38.1/2) * 1e-03
length = 2 * np.pi * initial_radius
theta = np.linspace(0, np.pi/2, 10000)
sin_theta = np.sin(theta)
desired_slope = 90
slopes = []
weight = 1000


#==========================================================================

#                          Helper Functions

# Compute Frenet-Serret frame from three arrays of data, (x,y and z)
# Method to compute the curvature
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

# Method to compute the distributed macrobend loss as a function of the curvature radius (1/k)
def curvature_loss_exp(curvature):
    a1 = 1525.965951223269
    b1 = -869.0646471499498
    a2 = 1525.9295489081296
    b2 = -869.0650046536391

    return a1 * np.exp(b1 * (1 / curvature)) + a2 * np.exp(b2 * (1 / curvature))

# Function to compute the elliptic integral
def calculate_elliptic_integral(eccentricity, sin_theta, theta):
    return trapz(np.sqrt(1 - eccentricity**2 * sin_theta**2), theta)

# Calculating a and b of an ellipse
def calculate_a_b(length, elliptic_integral, eccentricity):
    b = (1 / 4) * length * np.sqrt(1 - eccentricity**2) / elliptic_integral
    a = b / np.sqrt(1 - eccentricity**2)
    return a, b

# Calculate the second eccentricity
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
# Optimizing delta
def objective(delta, verbose=False):

    totalLoss1 = []
    totalLoss2 = []
    two_a_value = []
    a1_value = []
    a2_value = []
    e2_value = []
    b1_value = []
    b2_value = []
    
    for eccentricity in np.linspace(0, 0.95, 500):

        elliptic_integral = ellipe(eccentricity**2)
        a1, b1 = calculate_a_b(length, elliptic_integral, eccentricity)
        a2 = ((2 * b1) + delta) / 2
        two_a2 = 2 * a2

        # Choosing range for two_a2
        if (two_a2 < 0.05 or two_a2 > 0.052):
            continue

        # Finding the eccentricity of the second ellipse 
        function_Ee = length / (4 * a2)
        
        if (function_Ee < 1 or function_Ee > np.pi / 2):
            break
        
        try:
            e2 = calculate_eccentricity(error_function, function_Ee)
        except Exception:
            continue        

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

    if (len(two_a_value) == 0 or len (totalLoss_sum)) == 0:
        return None, None, None, None, None, None, None, None, None, None

    totalLoss_sum = np.array(totalLoss1) + np.array(totalLoss2)
    two_a_value = np.array(two_a_value)

    if (len(two_a_value) != 0 and len(totalLoss_sum) != 0):
    # Compute the MSE between the sum of curvature losses and a straight line fit
        mse, slope_difference, calculated_slope, calculated_intercept = fit_and_evaluate_losses(
            two_a_value, totalLoss_sum, desired_slope
        )
        slopes.append(calculated_slope)


    # Check if the current delta gives a better fit
        combined_metric = mse + 1000 * slope_difference 
        if combined_metric < min_combined_metric:
            min_combined_metric = combined_metric
            best_delta = delta

    return best_delta, two_a_value, totalLoss_sum, calculated_slope, calculated_intercept, a1_value, b1_value, a2_value, b2_value, e2_value

# Function to fit a line (linear regression) and plot the final result
def plot_best_fit_line(x, y):
    x = x.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # Plot the filtered data
    fig3, ax3 = plt.subplots(figsize=(13, 8))
    ax3.plot(x, y, label='Sum of Curvature Losses [dB] (filtered)', color='green', linewidth=2)
    
    # Plotar the fitted line
    ax3.plot(x, y_pred, label='Fitted line', color='red', linestyle='--', linewidth=2)
    ax3.set_title('Sum of Curvature Losses vs 2a (With Fitted Line)')
    ax3.set_xlabel('2a [m]')
    ax3.set_ylabel('Sum Loss [dB]')
    ax3.grid(True)
    ax3.tick_params(axis='both', which='major')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Slope and intercept
    print(f"Slope: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"calculated_slope: {calculated_slope}")



#==========================================================================
#
#                       Calculating the two ellipses

best_delta, two_a_value, totalLoss_sum, calculated_slope, calculated_intercept, a1, b1, a2, b2, e2  = optimize_delta(deltas, length, theta, sin_theta, curvature_loss_exp, TNB_Flat, error_function, desired_slope)

# Calculate the line with the optimized delta
calculated_slope = float(calculated_slope[0])  
calculated_intercept = float(calculated_intercept[0])  
line_values = calculated_slope * two_a_value + calculated_intercept



#==========================================================================
#
#                               Results

print(f"slope:",calculated_slope)
# The best delta
print(f"The best delta is {best_delta}")

# Applying the metric
print(len(two_a_value), len(totalLoss_sum))
r2 = calculate_r2(two_a_value, totalLoss_sum)
print(f"R²: {r2}")

# Criar o gráfico
plt.figure(figsize=(10, 6))

# Plotar as listas
plt.plot(e2, a1, label="a1", marker="o")
plt.plot(e2, a2, label="a2", marker="s")
plt.plot(e2, b1, label="b1", marker="^")
plt.plot(e2, b2, label="b2", marker="d")

# Personalizações
plt.xlabel("Excentricidade da segunda elipse")
plt.ylabel("Valores de a e b")
plt.title("Comportamento de a e b")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Plot 2a vs sum of total losses
fig1, ax1 = plt.subplots(figsize=(13, 8))
ax1.plot(
    two_a_value, 
    totalLoss_sum, 
    label=f'Sum of Curvature Losses [dB] (optimized delta = {best_delta})', 
    color='blue',
    linewidth=2)
ax1.plot(
    two_a_value,
    line_values,
    label=f'Fitted Line (slope={calculated_slope:.2f}, intercept={calculated_intercept:.2f})',
    color='red',
    linestyle='--',
    linewidth=2
)
ax1.set_title('Sum of Curvature Losses vs 2a (Optimized)')
ax1.set_xlabel('2a [m]')
ax1.set_ylabel('Sum Loss [dB]')
ax1.grid(True)
plt.tight_layout()
ax1.tick_params(axis='both', 
                which='major')
plt.legend()
plt.show()



#ao inves de mse vamos usar o algoritmo de gradiente descendente para escolher o delta
