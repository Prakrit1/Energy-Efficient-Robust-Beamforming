from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.colors as mc
import colorsys

class PlotConfig:
    """Define common parameters for consistent plotting."""

    def __init__(
            self,
    ) -> None:

        self._pre_init()

        self.textwidth = 5.81

        plt.rc('axes', labelsize=2*10)
        plt.rc('axes', labelsize=2*9.13)
        plt.rc('xtick', labelsize=2*6.67)
        plt.rc('ytick', labelsize=2*6.67)
        plt.rc('legend', fontsize=2*9.13)

        plt.rc(
            'legend',
            framealpha=1.0,
            fancybox=False,
            columnspacing=1.0,
        )

        plt.rc(
            'xtick',
            direction='in',
        )
        plt.rc(
            'ytick',
            direction='in',
        )

        plt.rc('lines', linewidth=1.2)
        plt.rc('lines', markersize=6)
        plt.rc('errorbar', capsize=3)

        plt.rc(
            'grid',
            color='gainsboro',
            linewidth=0.4
        )
        plt.rc('axes.spines',
               top=False, right=False,

        )
        plt.rc('xtick.minor', visible=True)
        plt.rc('ytick.minor', visible=True)
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)

    def _pre_init(
            self,
    ) -> None:

        self.project_root_path = Path(__file__).parent.parent.parent
        self.plots_parent_path = Path(self.project_root_path, 'reports', 'figures')

        self.plots_parent_path.mkdir(parents=True, exist_ok=True)

        self.cp2: dict[str: str] = {
            'magenta': '#d01b88',
            'blue': '#254796',
            'green': '#307b3b',
            'gold': '#caa023',
            'white': '#ffffff',
            'black': '#000000',
        }

        self.cp3: dict[str: str] = {
            'red1': '#9d2246',
            'red2': '#d50c2f',
            'red3': '#f39ca9',
            'blue1': '#00326d',
            'blue2': '#0068b4',
            'blue3': '#89b4e1',
            'purple1': '#3b296a',
            'purple2': '#8681b1',
            'purple3': '#c7c1e1',
            'peach1': '#d45b65',
            'peach2': '#f4a198',
            'peach3': '#fbdad2',
            'orange1': '#f7a600',
            'orange2': '#fece43',
            'orange3': '#ffe7b6',
            'green1': '#008878',
            'green2': '#8acbb7',
            'green3': '#d6ebe1',
            'yellow1': '#dedc00',
            'yellow2': '#f6e945',
            'yellow3': '#fff8bd',
            'white': '#ffffff',
            'black': '#000000',
        }

        self.cp4 = {
            'brown': '#D3BBA7',
            'grey': '#C5CDD3',
            'red': '#ECC0C1',
            'orange': '#FFDCCC',
            'vanilla': '#FFF9EC',
            'mint': '#d8e2dc',
            'blue': '#C6DEF1',
            'white': '#ffffff',
            'black': '#000000',
        }

        self.cp5 = {
            'brown': '#E2CFC4',
            'orange': '#F7D9C4',
            'yellow': '#FAEDCB',
            'green': '#C9E4DE',
            'blue': '#C6DEF1',
            'purple': '#DBCDF0',
            'pink': '#F2C6DE',
            'red': '#F9C6C9',
            'light_grey': '#E2E2DF',
            'dark_grey': '#D2D2CF',
            'white': '#ffffff',
            'black': '#000000',
        }

def save_figures(
        plots_parent_path,
        plot_name,
        padding,
) -> None:
    """Save a pyplot fig in multiple formats."""

    pgf_path = Path(plots_parent_path, 'pgf')
    pdf_path = Path(plots_parent_path, 'pdf')
    eps_path = Path(plots_parent_path, 'eps')
    jpg_path = Path(plots_parent_path, 'jpg')
    png_path = Path(plots_parent_path, 'png')

    pgf_path.mkdir(exist_ok=True)
    pdf_path.mkdir(exist_ok=True)
    eps_path.mkdir(exist_ok=True)
    jpg_path.mkdir(exist_ok=True)
    png_path.mkdir(exist_ok=True)

    plt.savefig(
        Path(pgf_path, f'{plot_name}.pgf'),
        bbox_inches='tight',
        pad_inches=padding,
        transparent=True,
    )
    plt.savefig(
        Path(pdf_path, f'{plot_name}.pdf'),
        bbox_inches='tight',
        pad_inches=padding,
        dpi=800,
        transparent=True,
    )
    plt.savefig(
        Path(eps_path, f'{plot_name}.eps'),
        bbox_inches='tight',
        pad_inches=padding,
    )

    plt.savefig(
        Path(jpg_path, f'{plot_name}.jpg'),
        bbox_inches='tight',
        pad_inches=padding,
        dpi=200,
    )

    plt.savefig(
        Path(png_path, f'{plot_name}.png'),
        bbox_inches='tight',
        pad_inches=padding,
        dpi=200,
        transparent=True,
    )

def generic_styling(
        ax,
) -> None:
    """Generic styling used for most plots."""

    ax.set_axisbelow(True)
    ax.grid()
    ax.grid(
        which='minor',
        color='whitesmoke',
        linewidth=0.8 * plt.rcParams['grid.linewidth']
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax.tick_params(axis='y', which='minor', left=False)
    ax.tick_params(axis='y', which='major', left=False)
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='x', which='major', bottom=False)

def pt_to_inches(
        pt: float
) -> float:
    """Convert pt to inches."""

    return 0.01389 * pt

def plot_color_palette(
        color_palette: dict
) -> None:
    """Plot a color palette from the plotting config."""

    list_of_colors = list(color_palette.keys())

    plt.figure()
    for color_id in range(len(list_of_colors)):
        plt.barh(color_id, 10, color=color_palette[list_of_colors[color_id]])

    plt.yticks(range(len(list_of_colors)), list_of_colors)
    plt.grid(alpha=.25)
    plt.show()

def change_lightness(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    amount<1 -> lighter
    amount>1 -> darker

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
