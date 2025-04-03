from cycler import cycler
from matplotlib import pyplot as plt


def startup_plotting(font_size=14, line_width=1.5, output_dpi=600, tex_backend=True):
    """Edited from https://github.com/nackjaylor/formatting_tips-tricks/"""

    if tex_backend:
        try:
            plt.rcParams.update({
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                        })
        except:
            print("WARNING: LaTeX backend not configured properly. Not using.")
            plt.rcParams.update({"font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                        })
    
    # Default settings
    plt.rcParams.update({
        "lines.linewidth": line_width,
        
        "axes.grid" : True, 
        "axes.grid.which": "major",
        "axes.linewidth": 0.5,
        "axes.prop_cycle": cycler("color", [
            "#0072B2", "#E69F00", "#009E73", "#CC79A7", 
            "#56B4E9", "#D55E00", "#F0E442", "#000000"]),

        "errorbar.capsize": 2.5,
        
        "grid.linewidth": 0.25,
        "grid.alpha": 0.5,
        
        "legend.framealpha": 0.7,
        "legend.edgecolor": [1,1,1],
        
        "savefig.dpi": output_dpi,
        "savefig.format": 'pdf'
    })

    # Change default font sizes.
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('xtick', labelsize=0.8*font_size)
    plt.rc('ytick', labelsize=0.8*font_size)
    plt.rc('legend', fontsize=0.8*font_size)
