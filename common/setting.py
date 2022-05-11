import matplotlib

def rcParamsSetting(dpi):
    matplotlib.rcParams['axes.linewidth']=1.*(100/dpi)
    matplotlib.rcParams['axes.labelsize']=10*(100/dpi)
    matplotlib.rcParams['axes.titlesize']=10*(100/dpi)
    matplotlib.rcParams['xtick.labelsize']=10*(100/dpi)
    matplotlib.rcParams['ytick.labelsize']=10*(100/dpi)
    matplotlib.rcParams['xtick.major.size']=3.5*(100/dpi)
    matplotlib.rcParams['xtick.major.width']=0.8*(100/dpi)
    matplotlib.rcParams['xtick.minor.size']=2*(100/dpi)
    matplotlib.rcParams['xtick.minor.width']=0.6*(100/dpi)
    matplotlib.rcParams['ytick.major.size']=3.5*(100/dpi)
    matplotlib.rcParams['ytick.major.width']=0.8*(100/dpi)
    matplotlib.rcParams['ytick.minor.size']=2*(100/dpi)
    matplotlib.rcParams['ytick.minor.width']=0.6*(100/dpi)

