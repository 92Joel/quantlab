import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
from matplotlib import figure
import seaborn as sns
from functools import wraps
from .statistics import Statistics

def plotting_context(func):
    """
    Decorator to set plotting context during function call.
    """

    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop('set_context', True)
        if set_context:
            with context():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return call_w_context


def context(context='notebook', font_scale=1.2, rc=None):
    """
    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.
    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by factor font_scale.
    rc : dict, optional
        Config flags.
        By default, {'lines.linewidth': 1.5,
                     'axes.facecolor': '0.995',
                     'figure.facecolor': '0.97'}
        is being used and will be added to any
        rc passed in, unless explicitly overriden.
    Returns
    -------
    seaborn plotting context
    Example
    -------
    >>> with quantlab.plotting.context(font_scale=2):
    >>>    quantlab.TearSheet.create_full_tear_sheet()
    See also
    --------
    For more information, see seaborn.plotting_context().
    """

    if rc is None:
        rc = {}

    rc_default = {'lines.linewidth': 1.5,
                  'axes.facecolor': '0.995',
                  'figure.facecolor': '0.2'}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale,
rc=rc)



    
