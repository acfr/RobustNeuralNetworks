#!/usr/bin/env python3


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import jax.numpy as jnp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_arm_results(sampled_data, test_output, train_dir, filename='3d_arm_results.pdf'):
    num_points = len(sampled_data['ytest'])
    colours = plt.cm.inferno(jnp.linspace(0, 1, num_points))


    with PdfPages(f'{train_dir}/{filename}') as pdf:
        # Page 1: Full 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')


        sc1 = ax.scatter(sampled_data['ytest'][:, 0],
                   sampled_data['ytest'][:, 1],
                   sampled_data['ytest'][:, 2],
                   c=colours, cmap=plt.cm.inferno, marker='o', label='Ground Truth')


        sc2 = ax.scatter(test_output[:, 0],
                   test_output[:, 1],
                   test_output[:, 2],
                   c=colours, cmap=plt.cm.inferno, marker='x', label='Model Predictions')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Arm Configuration Space')
        plt.legend()


        cbar = plt.colorbar(sc1, ax=ax, pad=0.1)
        cbar.set_label('Point Index')

        pdf.savefig()
        plt.close()

        # Page 2: Multiple views on a single page
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(2, 3)


        views = [
            (0, 0),    # Front view (XY plane)
            (90, 0),   # Side view (YZ plane)
            (0, 90),   # Top view (XZ plane)
            (45, 30),  # 3/4 view
            (135, 30), # Another 3/4 view
            (180, 30)  # Rear 3/4 view
        ]

        view_names = ['Front View', 'Side View', 'Top View',
                     '3/4 Front View', '3/4 Rear View', 'Rear View']

        # Create each subplot with a different viewing angle
        for i, ((azim, elev), name) in enumerate(zip(views, view_names)):
            row = i // 3
            col = i % 3

            ax = fig.add_subplot(gs[row, col], projection='3d')


            ax.scatter(sampled_data['ytest'][:, 0],
                      sampled_data['ytest'][:, 1],
                      sampled_data['ytest'][:, 2],
                      c=colours, cmap=plt.cm.inferno, marker='o', s=20)


            ax.scatter(test_output[:, 0],
                      test_output[:, 1],
                      test_output[:, 2],
                      c=colours, cmap=plt.cm.inferno, marker='x', s=20)

            ax.view_init(elev=elev, azim=azim)
            ax.set_title(name)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')


            max_range = np.array([
                np.max(sampled_data['ytest'][:, 0]) - np.min(sampled_data['ytest'][:, 0]),
                np.max(sampled_data['ytest'][:, 1]) - np.min(sampled_data['ytest'][:, 1]),
                np.max(sampled_data['ytest'][:, 2]) - np.min(sampled_data['ytest'][:, 2])
            ]).max() / 2.0

            mid_x = (np.max(sampled_data['ytest'][:, 0]) + np.min(sampled_data['ytest'][:, 0])) * 0.5
            mid_y = (np.max(sampled_data['ytest'][:, 1]) + np.min(sampled_data['ytest'][:, 1])) * 0.5
            mid_z = (np.max(sampled_data['ytest'][:, 2]) + np.min(sampled_data['ytest'][:, 2])) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        legend_ax = fig.add_subplot(gs[:, -1])
        legend_ax.axis('off')

        o_marker = plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=plt.cm.inferno(0.5), markersize=10)
        x_marker = plt.Line2D([0], [0], marker='x', color=plt.cm.inferno(0.5), markersize=10)

        legend_ax.legend([o_marker, x_marker],
                         ['Ground Truth', 'Model Predictions'],
                         loc='center')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Page 3: Three orthogonal 2D projections
        fig = plt.figure(figsize=(15, 5))

        # XY
        ax1 = fig.add_subplot(131)
        ax1.scatter(sampled_data['ytest'][:, 0], sampled_data['ytest'][:, 1],
                   c=colours, cmap=plt.cm.inferno, marker='o', alpha=0.7)
        ax1.scatter(test_output[:, 0], test_output[:, 1],
                   c=colours, cmap=plt.cm.inferno, marker='x', alpha=0.7)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('XY Projection')
        ax1.set_aspect('equal')

        # XZ
        ax2 = fig.add_subplot(132)
        ax2.scatter(sampled_data['ytest'][:, 0], sampled_data['ytest'][:, 2],
                   c=colours, cmap=plt.cm.inferno, marker='o', alpha=0.7)
        ax2.scatter(test_output[:, 0], test_output[:, 2],
                   c=colours, cmap=plt.cm.inferno, marker='x', alpha=0.7)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('XZ Projection')
        ax2.set_aspect('equal')

        # YZ
        ax3 = fig.add_subplot(133)
        ax3.scatter(sampled_data['ytest'][:, 1], sampled_data['ytest'][:, 2],
                   c=colours, cmap=plt.cm.inferno, marker='o', alpha=0.7)
        ax3.scatter(test_output[:, 1], test_output[:, 2],
                   c=colours, cmap=plt.cm.inferno, marker='x', alpha=0.7)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        ax3.set_title('YZ Projection')
        ax3.set_aspect('equal')

        o_marker = plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=plt.cm.inferno(0.5), markersize=10)
        x_marker = plt.Line2D([0], [0], marker='x', color=plt.cm.inferno(0.5), markersize=10)

        fig.legend([o_marker, x_marker],
                  ['Ground Truth', 'Model Predictions'],
                  loc='lower center', ncol=2)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Page 4: Error visualization
        fig = plt.figure(figsize=(15, 10))

        errors = jnp.sqrt(jnp.sum((test_output - sampled_data['ytest'])**2, axis=1))

        sorted_error_indices = jnp.argsort(errors)
        sorted_errors = errors[sorted_error_indices]
        sorted_ground_truth = sampled_data['ytest'][sorted_error_indices]
        sorted_predictions = test_output[sorted_error_indices]

        error_colors = plt.cm.viridis(sorted_errors / jnp.max(sorted_errors))

        ax1 = fig.add_subplot(121, projection='3d')
        sc = ax1.scatter(sorted_ground_truth[:, 0],
                        sorted_ground_truth[:, 1],
                        sorted_ground_truth[:, 2],
                        c=sorted_errors, cmap='viridis', s=30)

        step = max(1, len(sorted_ground_truth) // 50)  # Show at most 50 lines
        for i in range(0, len(sorted_ground_truth), step):
            ax1.plot([sorted_ground_truth[i, 0], sorted_predictions[i, 0]],
                    [sorted_ground_truth[i, 1], sorted_predictions[i, 1]],
                    [sorted_ground_truth[i, 2], sorted_predictions[i, 2]],
                    'k-', alpha=0.3)

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Error Visualization')
        cbar = plt.colorbar(sc, ax=ax1)
        cbar.set_label('Error Magnitude')

        ax2 = fig.add_subplot(122)
        ax2.hist(errors, bins=30, alpha=0.7, color='blue')
        ax2.set_xlabel('Error Magnitude')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')

        mean_error = jnp.mean(errors)
        median_error = jnp.median(errors)
        max_error = jnp.max(errors)

        stats_text = (f'Mean Error: {mean_error:.4f}\n'
                     f'Median Error: {median_error:.4f}\n'
                     f'Max Error: {max_error:.4f}')

        ax2.text(0.95, 0.95, stats_text,
                transform=ax2.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.tight_layout()
        pdf.savefig()
        plt.close()


def plot_3d_arm_results_delta(sampled_data, test_output, train_dir, filename='3d_arm_results.pdf'):
    num_points = len(sampled_data['ytest'])
    colours = plt.cm.inferno(jnp.linspace(0, 1, num_points))

    with PdfPages(f'{train_dir}/{filename}') as pdf:
        # Page 1: Full 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Draw lines between corresponding points
        for i in range(num_points):
            ax.plot([sampled_data['ytest'][i, 0], test_output[i, 0]],
                    [sampled_data['ytest'][i, 1], test_output[i, 1]],
                    [sampled_data['ytest'][i, 2], test_output[i, 2]],
                    'gray', alpha=1.0, linewidth=3.0)

        sc1 = ax.scatter(sampled_data['ytest'][:, 0],
                   sampled_data['ytest'][:, 1],
                   sampled_data['ytest'][:, 2],
                   c='red', marker='o', label='Ground Truth')

        sc2 = ax.scatter(test_output[:, 0],
                   test_output[:, 1],
                   test_output[:, 2],
                   c='black', marker='x', label='Model Predictions')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Arm Configuration Space')
        plt.legend()

        # cbar = plt.colorbar(sc1, ax=ax, pad=0.1)
        # cbar.set_label('Point Index')

        pdf.savefig()
        plt.close()


        # View 1: X-Y plane (top view)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.scatter(sampled_data['ytest'][:, 0], sampled_data['ytest'][:, 1], c='red', marker='o', label='Ground Truth')
        ax.scatter(test_output[:, 0], test_output[:, 1], c='black', marker='x', label='Model Predictions')

        # Draw lines between corresponding points
        for i in range(num_points):
            ax.plot([sampled_data['ytest'][i, 0], test_output[i, 0]],
                    [sampled_data['ytest'][i, 1], test_output[i, 1]],
                    'gray', alpha=0.3, linewidth=3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Top View (X-Y plane)')
        plt.legend()
        pdf.savefig()
        plt.close()

        # View 2: X-Z plane (front view)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.scatter(sampled_data['ytest'][:, 0], sampled_data['ytest'][:, 2], c='red', marker='o', label='Ground Truth')
        ax.scatter(test_output[:, 0], test_output[:, 2], c='black', marker='x', label='Model Predictions')

        # Draw lines between corresponding points
        for i in range(num_points):
            ax.plot([sampled_data['ytest'][i, 0], test_output[i, 0]],
                    [sampled_data['ytest'][i, 2], test_output[i, 2]],
                    'gray', alpha=1.0, linewidth=3)

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title('Front View (X-Z plane)')
        plt.legend()
        pdf.savefig()
        plt.close()

        # View 3: Y-Z plane (side view)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.scatter(sampled_data['ytest'][:, 1], sampled_data['ytest'][:, 2], c='red', marker='o', label='Ground Truth')
        ax.scatter(test_output[:, 1], test_output[:, 2], c='black', marker='x', label='Model Predictions')

        # Draw lines between corresponding points
        for i in range(num_points):
            ax.plot([sampled_data['ytest'][i, 1], test_output[i, 1]],
                    [sampled_data['ytest'][i, 2], test_output[i, 2]],
                    'gray', alpha=1.0, linewidth=3)

        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_title('Side View (Y-Z plane)')
        plt.legend()
        pdf.savefig()
        plt.close()

        # Alternative 3D views
        for elev, azim, view_name in [(30, 45, 'Perspective 1'),
                                      (0, 0, 'Front'),
                                      (0, 90, 'Side'),
                                      (90, 0, 'Top')]:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Draw lines between corresponding points
            for i in range(num_points):
                ax.plot([sampled_data['ytest'][i, 0], test_output[i, 0]],
                        [sampled_data['ytest'][i, 1], test_output[i, 1]],
                        [sampled_data['ytest'][i, 2], test_output[i, 2]],
                        'gray', alpha=1.0, linewidth=3)

            ax.scatter(sampled_data['ytest'][:, 0],
                       sampled_data['ytest'][:, 1],
                       sampled_data['ytest'][:, 2],
                       c='red', marker='o', label='Ground Truth')

            ax.scatter(test_output[:, 0],
                       test_output[:, 1],
                       test_output[:, 2],
                       c='black', marker='x', label='Model Predictions')

            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D View - {view_name}')
            plt.legend()
            pdf.savefig()
            plt.close()
