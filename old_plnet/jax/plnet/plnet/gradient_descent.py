# add functions for gradient descent related

import jax
import jax.numpy as jnp
from scipy.integrate import RK45

'''
check model gradient descent
# '''
# def get_gradient_descent_step( network_output, learning_rate):
# 	@jax.jit
# 	def gradient_descent_step(point):
# 		"""
# 		Perform a single step of gradient descent.
		
# 		Args:
# 			point (jax.numpy.ndarray): The current point in the input space.
# 			network_output (function): A function that takes an input point and returns the network output.
# 			learning_rate (float): The step size for gradient descent.
		
# 		Returns:
# 			jax.numpy.ndarray: The updated point after taking one gradient descent step.
# 		"""
# 		grad_fn = jax.grad(network_output)
# 		grad = grad_fn(point)
# 		return point - learning_rate * grad
      
# 	return gradient_descent_step


def get_gradient_descent_step(network_output, initial_step_size=1.0, 
            beta=0.5, tolerance=1e-6, max_iter=10,
            is_normalized = True):
    @jax.jit
    def gradient_descent_step(point):
        """
        Perform a single step of gradient descent with variable step size and normalized gradient,
        using a jitable line search.
        
        Args:
            point (jax.numpy.ndarray): The current point in the input space.
            network_output (function): A function that takes an input point and returns the network output.
            initial_step_size (float): The initial step size for line search.
            beta (float): Factor to decrease step size during line search (0 < beta < 1).
            tolerance (float): Tolerance for step size stopping criterion.
            max_iter (int): Maximum number of line search iterations.
        
        Returns:
            jax.numpy.ndarray: The updated point after taking one gradient descent step.
        """
        grad_fn = jax.grad(network_output)
        grad = grad_fn(point)
        
        # Normalize the gradient to have unit length
        grad_norm = jnp.linalg.norm(grad)
        normalized_grad = jax.numpy.where( grad_norm > 0 & is_normalized,  grad / grad_norm, grad)
        # normalized_grad = grad / grad_norm if grad_norm > 0 else grad
        # jax.debug.print("start of gradeient step with grad: {}", normalized_grad)

        # Define initial state for line search
        state = {
            'step_size': initial_step_size,
            'iteration': 0,
            'current_point': point,
            'grad_direction': normalized_grad
        }

        def cond(state):
            """Condition for continuing line search."""
            # jax.debug.print("original: {} \nnew: {}\ndiff: {}\ngap: {}", network_output(state['current_point']), 
            #                 network_output(state['current_point'] - state['step_size'] * state['grad_direction']),
            #                 network_output(state['current_point']) - network_output(state['current_point'] - state['step_size'] * state['grad_direction']),
            #                 tolerance * state['step_size'] * jnp.dot(state['grad_direction'], state['grad_direction']))

            return state['iteration'] < max_iter

        def body(state):
            """Body of the line search loop."""
            new_point = state['current_point'] - state['step_size'] * state['grad_direction']

            new_point = jnp.where( network_output(state['current_point']) - network_output(new_point) > 
                 tolerance * state['step_size'] * jnp.dot(state['grad_direction'], state['grad_direction']), 
                 new_point, state['current_point'])

            # jax.debug.print("newpoint: {}", new_point)
            return {
                'step_size': state['step_size'] * beta,
                'iteration': state['iteration'] + 1,
                'current_point': new_point,
                'grad_direction': state['grad_direction']
            }

        final_state = jax.lax.while_loop(cond, body, state)
        return final_state['current_point']
    
    return gradient_descent_step


def get_gradient_descent_path(network_output, learning_rate=0.01, num_steps=150, 
                                is_rk45 = False, is_normalized = True):
    """
    Performs gradient descent on the network's output starting from the given point, using JAX's `lax.scan`.
    
    Args:
        point (jax.numpy.ndarray): Initial point in the input space.
        network_output (function): A function that takes an input point and returns the network output.
        learning_rate (float): The step size for gradient descent.
        num_steps (int): The number of gradient descent steps to take.
    
    Returns:
        a jited function to do gradient descent
        eg, gradient_descent_path = get_gradient_descent_path(network_output, learning_rate, num_steps)
            path = gradient_descent_path(point)
    """
    if is_rk45 == False:
        gradient_descent_step = get_gradient_descent_step(network_output, 
                                initial_step_size = learning_rate, is_normalized=is_normalized)
        
        # keep decrease gradient step until the step do help decrease 
        # if the gradient descent works, update the 
        @jax.jit
        def gradient_descent_path(point):
            
            def step_function(point, _):
                new_point = gradient_descent_step(point)
                return new_point, new_point  # Return new point as carry and output

            # Run the scan over the number of steps
            _, path = jax.lax.scan(step_function, point, None, length=num_steps)
            
            # Include the initial point in the path
            path = jnp.vstack([point, path])  # Stack the initial point with the computed path
            
            return path
        return gradient_descent_path
    else:
        # ----------------------------------------------------------------------------------------
        # issue here - todo: the rk45 fail to take it as a function and actaully run 
        def gradient_descent_path_rk45(point, tolerance=1e-6, max_iter=10):
            """
            Generate a trajectory using adaptive gradient descent via RK45 integration.

            Args:
                point (numpy.ndarray): Initial point in the input space.
                network_output (function): A function that takes an input point and returns the network output.
                num_steps (int): Number of steps or points in the trajectory to record.
                tolerance (float): Tolerance for adaptive step-size termination.
                max_iter (int): Maximum number of RK45 integration steps.

            Returns:
                numpy.ndarray: A trajectory of points representing the path of descent.
            """
            # Compute the gradient function for the network output
            def gradient_flow(t, y):
                grad = jax.grad(network_output)(jnp.array(y))
                return -grad

            # Initialize RK45 solver for gradient descent
            solver = RK45(gradient_flow, t0=0, y0=point, t_bound=max_iter, rtol=tolerance, atol=tolerance)

            # Collect the trajectory points
            trajectory = [point]
            for _ in range(num_steps - 1):
                if solver.status == 'running':
                    solver.step()
                    trajectory.append(solver.y)  # Append the current point from the solver
                    jax.debug.print("at point: {}", solver.y)
                else:
                    break  # Stop if the solver is done

            jax.debug.print("trajectory from rk45: {}", trajectory)

            return jnp.vstack(trajectory)  # Stack points to form the trajectory
        return gradient_descent_path_rk45


def generate_sequence(points, step, step_size = 0.1):
    """
    Generate sequences by multiplying each pair of points by exp(-t), 
    where t ranges from 0 to m.
    
    Args:
        points: An (n x 2) array of n points.
        m: An integer representing the upper limit for t.
    
    Returns:
        A (n x (m+1) x 2) array where each point is expanded into a sequence 
        of length (m+1) with shape mx2.
    """
    # Create a sequence of t values from 0 to m
    t_values = jnp.arange(step+1) * step_size  # Shape: (m+1)
    
    # Compute the exponential values exp(-t) for t in [0, m]
    exp_t = jnp.exp(-t_values)  # Shape: (m+1)
    
    # Reshape exp_t for broadcasting across points
    exp_t = exp_t[:, None]  # Shape: (m+1, 1)
    
    # Multiply each point by the exp(-t) sequence
    sequences = points[:, None, :] * exp_t  # Shape: (n, m+1, 2)
    
    return sequences


def generate_path_in_gmap_space(points, gmap_fn, zero_point, num_steps, step_size = 0.1):
    '''
    take points in x space (nx2)
    map them to y space (nx2)
    generate the path to zero point in y space (nxstepx2)
    return the path in m space (nxstepx2)

    run inverse and plot after that
    points_y = generate_path_in_gmap_space(points, 
                        lambda point: model_pl.apply(params_pl, point, method=model_pl.gmap), 
                        jnp.array(zero_point), num_steps)

    for path in points_y:
        # convert back to x space
        path = inverse_func(path)
        plot_gradient_descent_path(ax, path, xlim=x_range, ylim=y_range)
    '''
    g_opt_x = gmap_fn(zero_point)

    # g(x_0)-g(x_ox*pt) nx2
    points = gmap_fn(points) - g_opt_x

    # x by exp term nx(step)x2 
    points_exp = generate_sequence(points=points, step=num_steps, step_size=step_size)

    # g(x) in y space nx(step)x2 
    points_y = points_exp + g_opt_x

    return points_y

def get_generate_trajectory(f, steps=500, dt=0.01):
    @jax.jit
    def generate_trajectory( x0, y0):
        """
        Generate a trajectory based on the dynamics system f(x, y).
        
        Parameters:
        - f: function representing the dynamics (x_dot, y_dot) = f(x, y)
        - x0, y0: initial position (float, float)
        - steps: number of steps in the trajectory (default: 500)
        - dt: timestep for integration (default: 0.01)
        
        Returns:
        - trajectory: an array of shape (steps, 2) representing the trajectory over time
        """
        trajectory = jnp.zeros((steps, 2))
        trajectory = trajectory.at[0].set(jnp.array([x0, y0]))
        
        # def step_fn(carry, _):
        #     pos = carry
        #     x, y = pos
        #     x_dot, y_dot = f(x, y)
        #     new_pos = pos + dt * jnp.array([x_dot, y_dot])
        #     return new_pos, new_pos
        
        @jax.jit
        def rk4_step(carry, _):
            """
            Perform a single Runge-Kutta 4th order (RK4) step using JAX.
            
            Args:
                x: The current state, a JAX array.
                dt: Time step size.
                
            Returns:
                x_next: The next state after one RK4 step.
            """
            x = carry
            k1 = f(x)
            k2 = f( x + (dt / 2) * k1)
            k3 = f( x + (dt / 2) * k2)
            k4 = f( x + dt * k3)
            
            # jax.debug.print("x dim: {}\n x_k1 dim: {}\n k1 dim: {}\n x_k2 dim: {}\n k2 dim: {}\n x_k3 dim: {}\n k3 dim: {}\nk4 dim: {}",
            #                 jnp.shape(x), jnp.shape(x + (dt / 2) * k1), jnp.shape(k1),
            #                 jnp.shape(x + (dt / 2) * k2), jnp.shape(k2),
            #                 jnp.shape(x + dt * k3), jnp.shape(k3),
            #                 jnp.shape(k4))
            
            # jax.debug.print("x: {}\n x_k1: {}\n k1: {}\n x_k2: {}\n k2: {}\n x_k3: {}\n k3: {}\nk4: {}",
            #                 x, x + dt / 2 * k1, k1,
            #                 x + dt / 2 * k2, k2,
            #                 x + dt * k3, k3,
            #                 k4)
            x_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            return x_next, x_next
        
        _, trajectory = jax.lax.scan(rk4_step, jnp.array([[x0, y0]]), jnp.arange(steps))
        return trajectory
    return generate_trajectory