#!/usr/bin/env python

import os, sys, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import pyqtSignal,Qt
from PyQt5.QtWidgets import QApplication,\
                            QPushButton,\
                            QWidget,\
                            QHBoxLayout,\
                            QVBoxLayout,\
                            QGridLayout,\
                            QLabel,\
                            QLineEdit,\
                            QTabWidget,\
                            QTabBar,\
                            QGroupBox,\
                            QDialog,\
                            QTableWidget,\
                            QTableWidgetItem,\
                            QInputDialog,\
                            QMessageBox,\
                            QComboBox,\
                            QShortcut,\
                            QFileDialog,\
                            QCheckBox,\
                            QRadioButton,\
                            QHeaderView,\
                            QSlider,\
                            QSpinBox,\
                            QDoubleSpinBox
from keras import models,layers
from scipy import interpolate

# Setting
base_path = os.path.abspath(os.path.dirname(sys.argv[0]))
background_path = base_path + '/images/insideKSTAR.jpg'
lstm_model_path = base_path + '/weights/lstm/'
nn_model_path = base_path + '/weights/nn'
bpw_model_path = base_path + '/weights/bpw'
k2rz_model_path = base_path + '/weights/k2rz'
max_models = 5
max_shape_models = 1
decimals = np.log10(200)
dpi = 1
plot_length = 40
year_in = 2021
ec_freq = 105.e9
steady_model = False

# Matplotlib rcParams setting
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

# Wall in KSTAR
Rwalls = np.array([1.265, 1.608, 1.683, 1.631, 1.578, 1.593, 1.626, 2.006,
                   2.233, 2.235, 2.263, 2.298, 2.316, 2.316, 2.298, 2.263,
                   2.235, 2.233, 2.006, 1.626, 1.593, 1.578, 1.631, 1.683,
                   1.608, 1.265, 1.265
                   ])
Zwalls = np.array([1.085, 1.429, 1.431, 1.326, 1.32, 1.153, 1.09, 0.773,
                   0.444, 0.369, 0.31, 0.189, 0.062, -0.062, -0.189, -0.31,
                   -0.369, -0.444, -0.773, -1.09, -1.153, -1.32, -1.326, -1.431,
                   -1.429, -1.085, 1.085
                   ])

# Inputs
input_params = ['Ip [MA]','Bt [T]','GW.frac. [-]',\
                'Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]',\
                'Pec2 [MW]','Pec3 [MW]','Zec2 [cm]','Zec3 [cm]',\
                'In.Mid. [m]','Out.Mid. [m]','Elon. [-]','Up.Tri. [-]','Lo.Tri [-]']
input_mins = [0.3,1.5,0.2, 0.0, 0.0, 0.0, 0.0,0.0,-10,-10, 1.265,2.18,1.6,0.1,0.5 ]
input_maxs = [0.8,2.7,0.6, 1.75,1.75,1.5, 0.8,0.8, 10, 10, 1.36, 2.29,2.0,0.5,0.9 ]
input_init = [0.5,1.8,0.4, 1.5, 0.0, 0.0, 0.0,0.0,0.0,0.0, 1.34, 2.22,1.7,0.3,0.75]

# Outputs
output_params0 = ['betan','q95','q0','li']
output_params1 = ['betap','wmhd']
output_params2 = ['betan','betap','h89','h98','q95','q0','li','wmhd']

def i2f(i,decimals=decimals):
    return float(i/10**decimals)

def f2i(f,decimals=decimals):
    return int(f*10**decimals)

class KSTARWidget(QDialog):
    def __init__(self, parent=None):
        super(KSTARWidget, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        
        # Initial condition
        self.first = True
        self.time = np.linspace(-0.1*(plot_length-1),0,plot_length)
        self.outputs = {}
        self.outputs['betan'] = [1.4035932701005382]
        self.outputs['betap'] = [1.0824991083280546]
        self.outputs['h89'] = [1.9199754370035778]
        self.outputs['h98'] = [1.2875278961044707]
        self.outputs['q95'] = [4.445880212274074]
        self.outputs['q0'] = [1.3098279277874445]
        self.outputs['li'] = [1.1197781355250758]
        self.outputs['wmhd'] = [186764.7911504754]
        self.x = np.zeros([10,21])

        # Load models
        self.k2rz = k2rz(n_models=max_shape_models)
        if steady_model:
            self.kstar_nn = kstar_nn(n_models=max_models)
        else:
            self.kstar_nn = kstar_nn(n_models=1)
            self.kstar_lstm = kstar_lstm(n_models=max_models)
        self.bpw_nn = bpw_nn(n_models=max_models)

        # Top layout
        topLayout = QHBoxLayout()
        
        nModelLabel = QLabel('# of models:')
        self.nModelBox = QSpinBox()
        self.nModelBox.setMinimum(1)
        self.nModelBox.setMaximum(max_models)
        self.nModelBox.setValue(1)
        self.resetModelNumber()
        self.nModelBox.valueChanged.connect(self.resetModelNumber)
        
        self.rtRunPushButton = QPushButton('Run')
        self.rtRunPushButton.setCheckable(True)
        self.rtRunPushButton.setChecked(True)
        self.rtRunPushButton.clicked.connect(self.reCreateOutputBox)

        self.shuffleModelPushButton = QPushButton('Shuffle models')
        self.shuffleModelPushButton.clicked.connect(self.shuffleModels)

        self.plotHeatLoadCheckBox = QCheckBox('Plot heat load')
        self.plotHeatLoadCheckBox.setChecked(True)
        self.plotHeatLoadCheckBox.stateChanged.connect(self.reCreateOutputBox)
        
        self.overplotCheckBox = QCheckBox('Overlap device')
        self.overplotCheckBox.setChecked(True)
        self.overplotCheckBox.stateChanged.connect(self.reCreateOutputBox)
        
        topLayout.addWidget(nModelLabel)
        topLayout.addWidget(self.nModelBox)
        topLayout.addWidget(self.rtRunPushButton)
        topLayout.addWidget(self.shuffleModelPushButton)
        topLayout.addWidget(self.plotHeatLoadCheckBox)
        topLayout.addWidget(self.overplotCheckBox)

        # Middle layout
        self.createInputBox()
        self.createOutputBox()

        # Bottom layout
        self.run1sButton = QPushButton('▶▶ 1s ▶▶')
        self.run1sButton.clicked.connect(self.relaxRun1s)
        self.run2sButton = QPushButton('▶▶ 2s ▶▶')
        self.run2sButton.clicked.connect(self.relaxRun2s)
        self.dumpButton = QPushButton('Dump outputs')
        self.dumpButton.clicked.connect(self.dumpOutput)

        # Main layout
        self.mainLayout = QGridLayout()

        self.mainLayout.addLayout(topLayout,0,0,1,2)
        self.mainLayout.addWidget(self.inputBox,1,0)
        self.mainLayout.addWidget(self.outputBox,1,1,1,2)
        self.mainLayout.addWidget(self.run1sButton,2,0)
        self.mainLayout.addWidget(self.run2sButton,2,1)
        self.mainLayout.addWidget(self.dumpButton,2,2)
        
        self.setLayout(self.mainLayout)

        self.setWindowTitle("KSTAR-NN simulator v0")
        self.tmp = 0

    def resetModelNumber(self):
        if steady_model:
            self.kstar_nn.nmodels = self.nModelBox.value()
        else:
            self.kstar_lstm.nmodels = self.nModelBox.value()
        self.bpw_nn.nmodels = self.nModelBox.value()

    def createInputBox(self):
        self.inputBox = QGroupBox('Input parameters')
        
        layout = QGridLayout()

        self.inputSliderDict = {}
        self.inputValueLabelDict = {}
        for input_param in input_params:
            idx = input_params.index(input_param)
            inputLabel = QLabel(input_param)
            self.inputSliderDict[input_param] = QSlider(Qt.Horizontal, self.inputBox)
            self.inputSliderDict[input_param].setMinimum(f2i(input_mins[idx]))
            self.inputSliderDict[input_param].setMaximum(f2i(input_maxs[idx]))
            self.inputSliderDict[input_param].setValue(f2i(input_init[idx]))
            self.inputSliderDict[input_param].valueChanged.connect(self.updateInputs)
            self.inputValueLabelDict[input_param] = QLabel(f'{self.inputSliderDict[input_param].value()/10**decimals:.3f}')
            self.inputValueLabelDict[input_param].setMinimumWidth(40)

            layout.addWidget(inputLabel,idx,0)
            layout.addWidget(self.inputSliderDict[input_param],idx,1)
            layout.addWidget(self.inputValueLabelDict[input_param],idx,2)

            #for widget in inputLabel,self.inputSliderDict[input_param],self.inputValueLabelDict[input_param]:
            #    widget.setMaximumWidth(30)
        self.runSlider = QSlider(Qt.Horizontal, self.inputBox)
        self.runSlider.setMinimum(0)
        self.runSlider.setMaximum(100)
        self.runSlider.setValue(0)
        self.runSlider.valueChanged.connect(self.updateInputs)
        self.runLabel = QLabel('0.1s ▶')

        layout.addWidget(QLabel('Run only'),len(input_params),0)
        layout.addWidget(self.runSlider,len(input_params),1)
        layout.addWidget(self.runLabel,len(input_params),2)

        self.inputBox.setLayout(layout)
        self.inputBox.setMaximumWidth(300)

    def updateInputs(self):
        for input_param in input_params:
            self.inputValueLabelDict[input_param].setText(f'{self.inputSliderDict[input_param].value()/10**decimals:.3f}')
        if self.rtRunPushButton.isChecked() and time.time()-self.tmp>0.05:
            self.reCreateOutputBox()
            self.tmp = time.time()

    def createOutputBox(self):
        self.outputBox = QGroupBox('Output')

        self.fig = plt.figure(figsize=(6*(100/dpi),4*(100/dpi)),dpi=dpi)
        self.plotPlasma()
        self.canvas = FigureCanvas(self.fig)

        self.layout = QGridLayout()
        self.layout.addWidget(self.canvas)

        self.outputBox.setLayout(self.layout)

    def reCreateOutputBox(self):
        self.outputBox = QGroupBox('       ')

        plt.clf()
        self.plotPlasma()
        self.canvas = FigureCanvas(self.fig)

        self.layout = QGridLayout()
        self.layout.addWidget(self.canvas)

        self.outputBox.setLayout(self.layout)
        self.mainLayout.replaceWidget(self.mainLayout.itemAtPosition(1,1).widget(),self.outputBox)

    def plotPlasma(self):
        # Predict plasma
        self.predictBoundary()
        if self.first or steady_model:
            self.predict0d(steady=True)
        else:
            self.predict0d(steady=False)
        ts = self.time[-len(self.outputs['betan']):]
        
        # Plot 2D view
        plt.subplot(1,2,1)
        plt.title('2D poloidal view')
        if self.overplotCheckBox.isChecked():
            self.plotBackground()
            plt.fill_between(self.rbdry,self.zbdry,color='b',alpha=0.2,linewidth=0.0)
        plt.plot(Rwalls,Zwalls,'k',linewidth=1.5*(100/dpi),label='Wall')
        plt.plot(self.rbdry,self.zbdry,'b',linewidth=2*(100/dpi),label='LCFS')
        #self.plotXpoints()
        if self.plotHeatLoadCheckBox.isChecked():
            self.plotHeatLoads()
        self.plotHeating()
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        if self.overplotCheckBox.isChecked():
            self.plotXpoints()
            plt.xlim([0.8,2.5])
            plt.ylim([-1.55,1.55])
        else:
            plt.axis('scaled')
            plt.grid(linewidth=0.5*(100/dpi))
            plt.legend(loc='center',fontsize=7.5*(100/dpi),markerscale=0.7,frameon=False)
            #plt.tight_layout(rect=(0.15,0.05,1.0,0.95))
        
        # Plot 0D evolution
        plt.subplot(4,2,2)
        plt.title('0D evolution')
        plt.plot(ts,self.outputs['betan'],'k',linewidth=2*(100/dpi),label='βN')
        plt.plot(ts,self.outputs['betap'],'b',linewidth=2*(100/dpi),label='βp')
        plt.grid(linewidth=0.5*(100/dpi))
        plt.legend(loc='upper left',fontsize=7.5*(100/dpi),frameon=False)
        plt.xlim([-0.1*plot_length-0.2,0.2])
        plt.ylim([0.5,3.0])
        plt.xticks(color='w')

        plt.subplot(4,2,4)
        plt.plot(ts,1.e-5*np.array(self.outputs['wmhd']),'k',linewidth=2*(100/dpi),label='10*Wmhd [MW]')
        plt.plot(ts,self.outputs['h89'],'b',linewidth=2*(100/dpi),label='H89')
        #plt.plot(ts,self.outputs['h98'],'b',linewidth=2*(100/dpi),label='H98')
        plt.grid(linewidth=0.5*(100/dpi))
        plt.legend(loc='upper left',fontsize=7.5*(100/dpi),frameon=False)
        plt.xlim([-0.1*plot_length-0.2,0.2])
        plt.ylim([1.5,4.5])
        plt.xticks(color='w')

        plt.subplot(4,2,6)
        plt.plot(ts,self.outputs['q95'],'k',linewidth=2*(100/dpi),label='q95')
        plt.plot(ts,self.outputs['q0'],'b',linewidth=2*(100/dpi),label='q0')
        plt.grid(linewidth=0.5*(100/dpi))
        plt.legend(loc='upper left',fontsize=7.5*(100/dpi),frameon=False)        
        plt.xlim([-0.1*plot_length-0.2,0.2])
        plt.ylim([1.0,None])
        plt.xticks(color='w')

        plt.subplot(4,2,8)
        plt.plot(ts,self.outputs['li'],'k',linewidth=2*(100/dpi),label='li')
        plt.plot(ts,2*np.array(self.outputs['betan'])*self.outputs['h89']/np.array(self.outputs['q95'])**2,'b',linewidth=2*(100/dpi),label='2*G')
        plt.grid(linewidth=0.5*(100/dpi))
        plt.legend(loc='upper left',fontsize=7.5*(100/dpi),frameon=False)
        plt.xlim([-0.1*plot_length-0.2,0.2])
        plt.ylim([0.4,1.2])

        plt.xlabel('Relative time [s]')
        plt.subplots_adjust(hspace=0.1)

        self.first = False

    def predictBoundary(self):
        ip = self.inputSliderDict[input_params[0]].value()/10**decimals
        bt = self.inputSliderDict[input_params[1]].value()/10**decimals
        bp = self.outputs['betap'][-1]
        rin = self.inputSliderDict[input_params[10]].value()/10**decimals
        rout = self.inputSliderDict[input_params[11]].value()/10**decimals
        k = self.inputSliderDict[input_params[12]].value()/10**decimals
        du = self.inputSliderDict[input_params[13]].value()/10**decimals
        dl = self.inputSliderDict[input_params[14]].value()/10**decimals

        self.k2rz.set_inputs(ip,bt,bp,rin,rout,k,du,dl)
        self.rbdry,self.zbdry = self.k2rz.predict(post=True)
        self.rx1 = self.rbdry[np.argmin(self.zbdry)]
        self.zx1 = np.min(self.zbdry)
        self.rx2 = self.rx1
        self.zx2 = -self.zx1

    def plotXpoints(self,mode=0):
        if mode==0:
            self.rx1 = self.rbdry[np.argmin(self.zbdry)]
            self.zx1 = np.min(self.zbdry)
            self.rx2 = self.rx1
            self.zx2 = -self.zx1
        plt.scatter([self.rx1,self.rx2],[self.zx1,self.zx2],marker='x',color='w',s=100*(100/dpi)**2,linewidths=2*(100/dpi),label='X-points')

    def plotHeatLoads(self,n=10,both_side=True):
        kinds = ['linear','quadratic'] #,'cubic']
        wallPath = Path(np.array([Rwalls,Zwalls]).T)
        idx1 = list(self.zbdry).index(self.zx1)
        for kind in kinds:
            f = interpolate.interp1d(self.rbdry[idx1-5:idx1],self.zbdry[idx1-5:idx1],kind=kind,fill_value='extrapolate')
            rsol1 = np.linspace(self.rbdry[idx1],np.min(Rwalls)+1.e-4,n)
            zsol1 = np.array([f(r) for r in rsol1])
            is_inside1 = wallPath.contains_points(np.array([rsol1,zsol1]).T)
            
            f = interpolate.interp1d(self.zbdry[idx1+5:idx1:-1],self.rbdry[idx1+5:idx1:-1],kind=kind,fill_value='extrapolate')
            zsol2 = np.linspace(self.zbdry[idx1],np.min(Zwalls)+1.e-4,n)
            rsol2 = np.array([f(z) for z in zsol2])
            is_inside2 = wallPath.contains_points(np.array([rsol2,zsol2]).T)
            if not np.all(zsol1[is_inside1]>self.zbdry[idx1+1]):
                plt.plot(rsol1[is_inside1],zsol1[is_inside1],'r',linewidth=1.5*(100/dpi))
            plt.plot(rsol2[is_inside2],zsol2[is_inside2],'r',linewidth=1.5*(100/dpi))
            if both_side:
                plt.plot(self.rbdry[idx1-4:idx1+4],-self.zbdry[idx1-4:idx1+4],'b',linewidth=2*(100/dpi),alpha=0.1)
                plt.plot(rsol1[is_inside1],-zsol1[is_inside1],'r',linewidth=1.5*(100/dpi),alpha=0.2)
                plt.plot(rsol2[is_inside2],-zsol2[is_inside2],'r',linewidth=1.5*(100/dpi),alpha=0.2)
        for kind in kinds:
            f = interpolate.interp1d(self.rbdry[idx1-5:idx1+1],self.zbdry[idx1-5:idx1+1],kind=kind,fill_value='extrapolate')
            rsol1 = np.linspace(self.rbdry[idx1],np.min(Rwalls)+1.e-4,n)
            zsol1 = np.array([f(r) for r in rsol1])
            is_inside1 = wallPath.contains_points(np.array([rsol1,zsol1]).T)

            f = interpolate.interp1d(self.zbdry[idx1+5:idx1-1:-1],self.rbdry[idx1+5:idx1-1:-1],kind=kind,fill_value='extrapolate')
            zsol2 = np.linspace(self.zbdry[idx1],np.min(Zwalls)+1.e-4,n)
            rsol2 = np.array([f(z) for z in zsol2])
            is_inside2 = wallPath.contains_points(np.array([rsol2,zsol2]).T)
            if not np.all(zsol1[is_inside1]>self.zbdry[idx1+1]):
                plt.plot(rsol1[is_inside1],zsol1[is_inside1],'r',linewidth=1.5*(100/dpi))
            plt.plot(rsol2[is_inside2],zsol2[is_inside2],'r',linewidth=1.5*(100/dpi))
            if both_side:
                plt.plot(rsol1[is_inside1],-zsol1[is_inside1],'r',linewidth=1.5*(100/dpi),alpha=0.2)
                plt.plot(rsol2[is_inside2],-zsol2[is_inside2],'r',linewidth=1.5*(100/dpi),alpha=0.2)
        plt.plot([self.rx1],[self.zx1],'r',linewidth=1*(100/dpi),label='Heat load')

    def plotBackground(self):
        img = plt.imread(background_path)
        plt.imshow(img,extent=[-1.6,2.45,-1.5,1.35])

    def plotHeating(self):
        pnb1a = self.inputSliderDict['Pnb1a [MW]'].value()/10**decimals
        pnb1b = self.inputSliderDict['Pnb1b [MW]'].value()/10**decimals
        pnb1c = self.inputSliderDict['Pnb1c [MW]'].value()/10**decimals
        pec2 = self.inputSliderDict['Pec2 [MW]'].value()/10**decimals
        pec3 = self.inputSliderDict['Pec3 [MW]'].value()/10**decimals
        zec2 = self.inputSliderDict['Zec2 [cm]'].value()/10**decimals
        zec3 = self.inputSliderDict['Zec3 [cm]'].value()/10**decimals
        bt = self.inputSliderDict['Bt [T]'].value()/10**decimals
        
        #rt1,rt2,rt3 = 1.48,1.73,1.23
        rt1,rt2,rt3 = 1.486,1.720,1.245
        #w,h = 0.114,0.42
        w,h = 0.13,0.45
        plt.fill_between([rt1-w/2,rt1+w/2],[-h/2,-h/2],[h/2,h/2],color='g',alpha=0.9 if pnb1a>0.5 else 0.3)
        plt.fill_between([rt2-w/2,rt2+w/2],[-h/2,-h/2],[h/2,h/2],color='g',alpha=0.9 if pnb1b>0.5 else 0.3)
        plt.fill_between([rt3-w/2,rt3+w/2],[-h/2,-h/2],[h/2,h/2],color='g',alpha=0.9 if pnb1c>0.5 else 0.3\
                         ,label='NBI')

        for ns in [1,2,3]:
            rs = 1.60219e-19*1.8*bt/(2.*np.pi*9.10938e-31*ec_freq)*ns
            if min(Rwalls)<rs<max(Rwalls):
                break
        dz = 0.05
        rpos,zpos = 2.449,0.35
        zres = zpos + (zec2/100-zpos)*(rs-rpos)/(1.8-rpos)
        plt.fill_between([rs,rpos],[zres-dz,zpos],[zres+dz,zpos],color='orange',alpha=0.9 if pec2>0.2 else 0.3)
        rpos,zpos = 2.451,-0.35
        zres = zpos + (zec3/100-zpos)*(rs-rpos)/(1.8-rpos)
        plt.fill_between([rs,rpos],[zres-dz,zpos],[zres+dz,zpos],color='orange',alpha=0.9 if pec3>0.2 else 0.3,\
                         label='ECH')

    def predict0d(self,steady=True):
        # Predict output_params0 (betan, q95, q0, li)
        if steady:
            x = np.zeros(17)
            idx_convert = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,10,2]
            for i in range(len(x)-1):
                x[i] = self.inputSliderDict[input_params[idx_convert[i]]].value()/10**decimals
            x[9],x[10] = 0.5*(x[9]+x[10]),0.5*(x[10]-x[9])
            x[14] = 1 if x[14]>1.265+1.e-4 else 0
            x[-1] = year_in
            y = self.kstar_nn.predict(x)
            for i in range(len(output_params0)):
                if len(self.outputs[output_params0[i]]) >= plot_length:
                    del self.outputs[output_params0[i]][0]
                elif len(self.outputs[output_params0[i]]) == 1:
                    self.outputs[output_params0[i]][0] = y[i]
                self.outputs[output_params0[i]].append(y[i])
            self.x[:,:len(output_params0)] = y
            self.x[:,len(output_params0):] = x
        else:
            self.x[:-1,len(output_params0):] = self.x[1:,len(output_params0):]
            idx_convert = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,10,2]
            for i in range(len(self.x[0])-1-4):
                self.x[-1,i+4] = self.inputSliderDict[input_params[idx_convert[i]]].value()/10**decimals
            self.x[-1,13],self.x[-1,14] = 0.5*(self.x[-1,13]+self.x[-1,14]),0.5*(self.x[-1,14]-self.x[-1,13])
            self.x[-1,18] = 1 if self.x[-1,18]>1.265+1.e-4 else 0
            y = self.kstar_lstm.predict(self.x)
            self.x[:-1,:len(output_params0)] = self.x[1:,:len(output_params0)]
            self.x[-1,:len(output_params0)] = y
            for i in range(len(output_params0)):
                if len(self.outputs[output_params0[i]]) >= plot_length:
                    del self.outputs[output_params0[i]][0]
                elif len(self.outputs[output_params0[i]]) == 1:
                    self.outputs[output_params0[i]][0] = y[i]
                self.outputs[output_params0[i]].append(y[i])

        # Predict output_params1 (betap, wmhd)
        x = np.zeros(8)
        idx_convert = [0,0,1,10,11,12,13,14]
        x[0] = self.outputs['betan'][-1]
        for i in range(1,len(x)):
            x[i] = self.inputSliderDict[input_params[idx_convert[i]]].value()/10**decimals
        x[3],x[4] = 0.5*(x[3]+x[4]),0.5*(x[4]-x[3])
        y = self.bpw_nn.predict(x)
        for i in range(len(output_params1)):
            if len(self.outputs[output_params1[i]]) >= plot_length:
                del self.outputs[output_params1[i]][0]
            elif len(self.outputs[output_params1[i]]) == 1:
                self.outputs[output_params1[i]][0] = y[i]
            self.outputs[output_params1[i]].append(y[i])

        # Estimate H factors (h89, h98)
        ip = self.inputSliderDict['Ip [MA]'].value()/10**decimals
        bt = self.inputSliderDict['Bt [T]'].value()/10**decimals
        fgw = self.inputSliderDict['GW.frac. [-]'].value()/10**decimals
        ptot = max(self.inputSliderDict['Pnb1a [MW]'].value()/10**decimals \
               + self.inputSliderDict['Pnb1b [MW]'].value()/10**decimals \
               + self.inputSliderDict['Pnb1c [MW]'].value()/10**decimals \
               + self.inputSliderDict['Pec2 [MW]'].value()/10**decimals \
               + self.inputSliderDict['Pec3 [MW]'].value()/10**decimals \
               , 1.e-1) # Not to diverge
        rin = self.inputSliderDict['In.Mid. [m]'].value()/10**decimals
        rout = self.inputSliderDict['Out.Mid. [m]'].value()/10**decimals
        k = self.inputSliderDict['Elon. [-]'].value()/10**decimals

        rgeo,amin = 0.5*(rin+rout),0.5*(rout-rin)
        ne = fgw*10*(ip/(np.pi*amin**2))
        m = 2.0 # Mass number

        tau89 = 0.038*ip**0.85*bt**0.2*ne**0.1*ptot**-0.5*rgeo**1.5*k**0.5*(amin/rgeo)**0.3*m**0.5
        tau98 = 0.0562*ip**0.93*bt**0.15*ne**0.41*ptot**-0.69*rgeo**1.97*k**0.78*(amin/rgeo)**0.58*m**0.19
        h89 = 1.e-6*self.outputs['wmhd'][-1]/ptot/tau89
        h98 = 1.e-6*self.outputs['wmhd'][-1]/ptot/tau98

        if len(self.outputs['h89']) >= plot_length:
            del self.outputs['h89'][0], self.outputs['h98'][0]
        elif len(self.outputs['h89']) == 1:
            self.outputs['h89'][0], self.outputs['h98'][0] = h89, h98

        self.outputs['h89'].append(h89)
        self.outputs['h98'].append(h98)

    def shuffleModels(self):
        np.random.shuffle(self.k2rz.models)
        if steady_model:
            np.random.shuffle(self.kstar_nn.models)
        else:
            np.random.shuffle(self.kstar_lstm.models)
        np.random.shuffle(self.bpw_nn.models)
        print('Models shuffled!')

    def relaxRun1s(self):
        for i in range(10-1):
            if self.first or steady_model:
                self.predict0d(steady=True)
            else:
                self.predict0d(steady=False)
        self.reCreateOutputBox()
        self.tmp = time.time()

    def relaxRun2s(self):
        for i in range(20-1):
            if self.first or steady_model:
                self.predict0d(steady=True)
            else:
                self.predict0d(steady=False)
        self.reCreateOutputBox()
        self.tmp = time.time()

    def dumpOutput(self):
        print('')
        print(f"Time [s]: {self.time[-len(self.outputs['betan']):]}")
        for output in output_params2:
            print(f'{output}: {self.outputs[output]}')


def r2_k(y_true, y_pred):
    #SS_res = K.sum(K.square(y_true - y_pred))
    #SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    #return ( 1 - SS_res/(SS_tot + epsilon) )
    return 1.0

class k2rz():
    def __init__(self,model_path=k2rz_model_path,n_models=10,ntheta=64,closed_surface=True,xpt_correction=True):
        self.nmodels = n_models
        self.ntheta = ntheta
        self.closed_surface = closed_surface
        self.xpt_correction = xpt_correction
        self.models = []
        for i in range(self.nmodels):
            self.models.append(models.load_model(model_path+f'/best_model{i}',custom_objects={'r2_k':r2_k}))
    
    def set_inputs(self,ip,bt,betap,rin,rout,k,du,dl):
        self.x = np.array([ip,bt,betap,rin,rout,k,du,dl])

    def predict(self,post=True):
        #print('predicting...')
        self.y = np.zeros(2*self.ntheta)
        for i in range(self.nmodels):
            self.y += self.models[i].predict(np.array([self.x]))[0]/self.nmodels
        rbdry,zbdry = self.y[:self.ntheta],self.y[self.ntheta:]
        if post:
            if self.xpt_correction:
                rgeo = 0.5*(max(rbdry)+min(rbdry))
                amin = 0.5*(max(rbdry)-min(rbdry))
                if self.x[6]<=self.x[7]:
                    rx = rgeo-amin*self.x[7]
                    zx = max(zbdry) - 2.*self.x[5]*amin
                    rx2 = rgeo-amin*self.x[6]
                    rbdry[np.argmin(zbdry)] = rx
                    zbdry[np.argmin(zbdry)] = zx
                    rbdry[np.argmax(zbdry)] = rx2
                if self.x[6]>=self.x[7]:
                    rx = rgeo-amin*self.x[6]
                    zx = min(zbdry) + 2.*self.x[5]*amin
                    rx2 = rgeo-amin*self.x[7]
                    rbdry[np.argmax(zbdry)] = rx
                    zbdry[np.argmax(zbdry)] = zx
                    rbdry[np.argmin(zbdry)] = rx2

            if self.closed_surface:
                rbdry = np.append(rbdry,rbdry[0])
                zbdry = np.append(zbdry,zbdry[0])

        return rbdry,zbdry

class kstar_lstm():
    def __init__(self,model_path=lstm_model_path,n_models=10):
        self.nmodels = n_models
        self.ymean = [1.30934765, 5.20082444, 1.47538417, 1.14439883]
        self.ystd = [0.74135689, 1.44731883, 0.56747578, 0.23018484]
        self.models = []
        for i in range(self.nmodels):
            self.models.append(models.Sequential())
            self.models[i].add(layers.BatchNormalization(input_shape=(10,21)))
            self.models[i].add(layers.LSTM(200,return_sequences=True))
            self.models[i].add(layers.BatchNormalization())
            self.models[i].add(layers.LSTM(200,return_sequences=False))
            self.models[i].add(layers.BatchNormalization())
            self.models[i].add(layers.Dense(200,activation='sigmoid'))
            self.models[i].add(layers.BatchNormalization())
            self.models[i].add(layers.Dense(4,activation='linear'))
            self.models[i].load_weights(model_path+f'/best_model{i}')

    def set_inputs(self,x):
        if len(np.shape(x)) == 3:
            self.x = np.array(x)
        else:
            self.x = np.array([x])

    def predict(self,x=None):
        if type(x) == type(np.zeros(1)):
            if len(np.shape(x)) == 3:
                self.x = np.array(x)
            else:
                self.x = np.array([x])
        self.y = np.zeros_like(self.ymean)
        for i in range(self.nmodels):
            self.y += (self.models[i].predict(self.x)[0]*self.ystd+self.ymean)/self.nmodels

        return self.y

class kstar_nn():
    def __init__(self,model_path=nn_model_path,n_models=10):
        self.nmodels = n_models
        self.ymean = [1.22379703, 5.2361062,  1.64438005, 1.12040048]
        self.ystd = [0.72255576, 1.5622809,  0.96563557, 0.23868018]
        self.models = []
        for i in range(self.nmodels):
            self.models.append(models.load_model(model_path+f'/best_model{i}',custom_objects={'r2_k':r2_k}))

    def set_inputs(self,x):
        self.x = np.array([x])

    def predict(self,x=None):
        if type(x) == type(np.zeros(1)):
            if len(np.shape(x)) == 2:
                self.x = x
            else:
                self.x = np.array([x])
        self.y = np.zeros(len(output_params0))
        for i in range(self.nmodels):
            self.y += (self.models[i].predict(self.x)[0]*self.ystd+self.ymean)/self.nmodels

        return self.y

class bpw_nn():
    def __init__(self,model_path=bpw_model_path,n_models=10):
        self.nmodels = n_models
        self.ymean = np.array([1.02158800e+00, 1.87408512e+05])
        self.ystd = np.array([6.43390272e-01, 1.22543529e+05])
        self.models = []
        for i in range(self.nmodels):
            self.models.append(models.load_model(model_path+f'/best_model{i}',custom_objects={'r2_k':r2_k}))
            
    def set_inputs(self,x):
        self.x = np.array([x])

    def predict(self,x=None):
        if type(x) == type(np.zeros(1)):
            if len(np.shape(x)) == 2:
                self.x = x
            else:
                self.x = np.array([x])
        self.y = np.zeros(len(output_params1))
        for i in range(self.nmodels):
            self.y += (self.models[i].predict(self.x)[0]*self.ystd+self.ymean)/self.nmodels

        return self.y


if __name__ == '__main__':
    app = QApplication([])
    window = KSTARWidget()
    window.show()
    app.exec()

