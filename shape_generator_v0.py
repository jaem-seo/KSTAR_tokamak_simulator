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
from keras import models
from scipy import interpolate

base_path = os.path.abspath(os.path.dirname(sys.argv[0]))
background_path = base_path + '/images/insideKSTAR.jpg'
k2rz_model_path = base_path + '/weights/k2rz/'
max_models = 5
decimals = np.log10(200)
dpi = 1

matplotlib.rcParams['axes.linewidth']=1.*(100/dpi)
matplotlib.rcParams['axes.labelsize']=10*(100/dpi)
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
#matplotlib.rcParams['axes.labelweight']='bold' 

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

def i2f(i,decimals=decimals):
    return float(i/10**decimals)

def f2i(f,decimals=decimals):
    return int(f*10**decimals)

class PBGWidget(QDialog):
    def __init__(self, parent=None):
        super(PBGWidget, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        
        topLayout = QHBoxLayout()
        nModelLabel = QLabel('# of models:')
        self.nModelBox = QSpinBox()
        self.nModelBox.setMinimum(1)
        self.nModelBox.setMaximum(max_models)
        self.nModelBox.setValue(max_models)
        self.plotXptCheckBox = QCheckBox('Plot X-points')
        self.plotHeatLoadCheckBox = QCheckBox('Plot heat load')
        self.plotBothSideCheckBox = QCheckBox('Plot both side')
        self.plotRTCheckBox = QCheckBox('RT plot')
        self.plotRTCheckBox.stateChanged.connect(self.rtButtonChanged)
        self.overplotCheckBox = QCheckBox('Overlap device')
        self.overplotCheckBox.stateChanged.connect(self.reCreatePlotBox)
        topLayout.addWidget(nModelLabel)
        topLayout.addWidget(self.nModelBox)
        topLayout.addWidget(self.plotXptCheckBox)
        topLayout.addWidget(self.plotHeatLoadCheckBox)
        topLayout.addWidget(self.plotBothSideCheckBox)
        topLayout.addWidget(self.plotRTCheckBox)
        topLayout.addWidget(self.overplotCheckBox)

        self.k2rz = k2rz(n_models=max_models)

        self.createInputBox()
        self.createPlotBox()

        self.runbutton = QPushButton('Run')
        self.runbutton.resize(10,10)
        self.runbutton.clicked.connect(self.reCreatePlotBox)
        self.dumpbutton = QPushButton('Dump')
        self.dumpbutton.clicked.connect(self.dumpBoundary)

        self.mainLayout = QGridLayout()

        self.mainLayout.addLayout(topLayout,0,0,1,2)
        self.mainLayout.addWidget(self.inputBox,1,0)
        self.mainLayout.addWidget(self.plotBox,1,1)
        self.mainLayout.addWidget(self.runbutton,2,0)
        self.mainLayout.addWidget(self.dumpbutton,2,1)
        self.setLayout(self.mainLayout)

        self.setWindowTitle("Plasma Boundary Model v0")
        self.tmp = 0

    def createInputBox(self):
        self.inputBox = QGroupBox('Input parameters')

        ipLabel = QLabel('Ip [MA]:')
        btLabel = QLabel('Bt [T]:')
        bpLabel = QLabel('βp:')
        self.ipBox = QDoubleSpinBox(self.inputBox)
        self.ipBox.setValue(0.5)
        self.btBox = QDoubleSpinBox(self.inputBox)
        self.btBox.setValue(1.8)
        self.betapBox = QDoubleSpinBox(self.inputBox)
        self.betapBox.setValue(2.0)

        kLabel = QLabel('Elon.:')
        self.kSlider = QSlider(Qt.Horizontal, self.inputBox)
        self.kSlider.setMinimum(f2i(1.6))
        self.kSlider.setMaximum(f2i(2.0))
        self.kSlider.setValue(f2i(1.8))
        self.kSlider.valueChanged.connect(self.updateInputs)
        self.kvlabel = QLabel(f'{self.kSlider.value()/10**decimals:.3f}')
        self.kvlabel.setMinimumWidth(40)
        duLabel = QLabel('Up.Tri.')
        self.duSlider = QSlider(Qt.Horizontal, self.inputBox)
        self.duSlider.setMinimum(f2i(0.1))
        self.duSlider.setMaximum(f2i(0.5))
        self.duSlider.setValue(f2i(0.3))
        self.duSlider.valueChanged.connect(self.updateInputs)
        self.duvlabel = QLabel(f'{self.duSlider.value()/10**decimals:.3f}')
        self.duvlabel.setMinimumWidth(40)
        dlLabel = QLabel('Lo.Tri.')
        self.dlSlider = QSlider(Qt.Horizontal, self.inputBox)
        self.dlSlider.setMinimum(f2i(0.5))
        self.dlSlider.setMaximum(f2i(0.9))
        self.dlSlider.setValue(f2i(0.75))
        self.dlSlider.valueChanged.connect(self.updateInputs)
        self.dlvlabel = QLabel(f'{self.dlSlider.value()/10**decimals:.3f}')
        self.dlvlabel.setMinimumWidth(40)
        rinLabel = QLabel('In.Mid. [m]')
        self.rinSlider = QSlider(Qt.Horizontal, self.inputBox)
        self.rinSlider.setMinimum(f2i(1.265))
        self.rinSlider.setMaximum(f2i(1.36))
        self.rinSlider.setValue(f2i(1.34))
        self.rinSlider.valueChanged.connect(self.updateInputs)
        self.rinvlabel = QLabel(f'{self.rinSlider.value()/10**decimals:.3f}')
        self.rinvlabel.setMinimumWidth(40)
        routLabel = QLabel('Out.Mid. [m]')
        self.routSlider = QSlider(Qt.Horizontal, self.inputBox)
        self.routSlider.setMinimum(f2i(2.18))
        self.routSlider.setMaximum(f2i(2.29))
        self.routSlider.setValue(f2i(2.22))
        self.routSlider.valueChanged.connect(self.updateInputs)
        self.routvlabel = QLabel(f'{self.routSlider.value()/10**decimals:.3f}')
        self.routvlabel.setMinimumWidth(40)

        layout = QGridLayout()
        layout.addWidget(ipLabel,0,0)
        layout.addWidget(self.ipBox,0,1)
        layout.addWidget(btLabel,1,0)
        layout.addWidget(self.btBox,1,1)
        layout.addWidget(bpLabel,2,0)
        layout.addWidget(self.betapBox,2,1)
        layout.addWidget(kLabel,3,0)
        layout.addWidget(self.kSlider,3,1)
        layout.addWidget(self.kvlabel,3,2)
        layout.addWidget(duLabel,4,0)
        layout.addWidget(self.duSlider,4,1)
        layout.addWidget(self.duvlabel,4,2)
        layout.addWidget(dlLabel,5,0)
        layout.addWidget(self.dlSlider,5,1)
        layout.addWidget(self.dlvlabel,5,2)
        layout.addWidget(rinLabel,6,0)
        layout.addWidget(self.rinSlider,6,1)
        layout.addWidget(self.rinvlabel,6,2)
        layout.addWidget(routLabel,7,0)
        layout.addWidget(self.routSlider,7,1)
        layout.addWidget(self.routvlabel,7,2)

        self.inputBox.setLayout(layout)

    def updateInputs(self):
        self.kvlabel.setText(f'{(self.kSlider.value()/10**decimals):.3f}')
        self.duvlabel.setText(f'{(self.duSlider.value()/10**decimals):.3f}')
        self.dlvlabel.setText(f'{(self.dlSlider.value()/10**decimals):.3f}')
        self.rinvlabel.setText(f'{(self.rinSlider.value()/10**decimals):.3f}')
        self.routvlabel.setText(f'{(self.routSlider.value()/10**decimals):.3f}')
        if self.plotRTCheckBox.isChecked() and time.time()-self.tmp>0.05:
            self.reCreatePlotBox()
            self.tmp = time.time()

    def rtButtonChanged(self):
        if self.plotRTCheckBox.isChecked():
            self.nModelBox.setValue(1)

    def createPlotBox(self):
        self.plotBox = QGroupBox('Output')

        self.fig = plt.figure(figsize=(2.5*(100/dpi),4*(100/dpi)),dpi=dpi)
        self.plotPlasma()
        self.canvas = FigureCanvas(self.fig)

        self.layout = QGridLayout()
        self.layout.addWidget(self.canvas)

        self.plotBox.setLayout(self.layout)

    def reCreatePlotBox(self):
        #self.mainLayout.removeWidget(self.plotBox)
        self.plotBox = QGroupBox(' ')

        plt.clf()
        self.plotPlasma()
        self.canvas = FigureCanvas(self.fig)

        self.layout = QGridLayout()
        self.layout.addWidget(self.canvas)

        self.plotBox.setLayout(self.layout)
        #self.mainLayout.addWidget(self.plotBox,1,1)
        self.mainLayout.replaceWidget(self.mainLayout.itemAtPosition(1,1).widget(),self.plotBox)

    def plotPlasma(self):
        rbdry,zbdry = self.predictBoundary()
        if self.overplotCheckBox.isChecked():
            self.plotBackground()
            plt.fill_between(rbdry,zbdry,color='b',alpha=0.2,linewidth=0.0)
        plt.plot(Rwalls,Zwalls,'k',linewidth=1.5*(100/dpi),label='Wall')
        plt.plot(rbdry,zbdry,'b',linewidth=2*(100/dpi),label='LCFS')
        if self.plotXptCheckBox.isChecked():
            self.plotXpoints()
        if self.plotHeatLoadCheckBox.isChecked():
            self.plotHeatLoads(both_side=self.plotBothSideCheckBox.isChecked())
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        if self.overplotCheckBox.isChecked():
            plt.xlim([0.8,2.5])
            plt.ylim([-1.55,1.55])
        else:
            plt.axis('scaled')
            plt.grid(linewidth=0.5*(100/dpi))
            plt.legend(loc='center',fontsize=7.5*(100/dpi),markerscale=0.7,frameon=False)
            plt.tight_layout(rect=(0.15,0.05,1.0,0.95))

    def predictBoundary(self):
        ip = self.ipBox.value()
        bt = self.btBox.value()
        bp = self.betapBox.value()
        rin = self.rinSlider.value()/10**decimals
        rout = self.routSlider.value()/10**decimals
        k = self.kSlider.value()/10**decimals
        du = self.duSlider.value()/10**decimals
        dl = self.dlSlider.value()/10**decimals
        self.k2rz.nmodels = self.nModelBox.value()
        self.k2rz.set_inputs(ip,bt,bp,rin,rout,k,du,dl)
        self.rbdry,self.zbdry = self.k2rz.predict(post=True)
        self.rx1 = self.rbdry[np.argmin(self.zbdry)]
        self.zx1 = np.min(self.zbdry)
        self.rx2 = self.rx1
        self.zx2 = -self.zx1
        return self.rbdry,self.zbdry

    def plotXpoints(self,mode=0):
        if mode==0:
            self.rx1 = self.rbdry[np.argmin(self.zbdry)]
            self.zx1 = np.min(self.zbdry)
            self.rx2 = self.rx1
            self.zx2 = -self.zx1
        plt.scatter([self.rx1,self.rx2],[self.zx1,self.zx2],marker='x',color='g',s=100*(100/dpi)**2,linewidths=2*(100/dpi),label='X-points')

    def plotHeatLoads(self,n=10,both_side=False):
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
        #plt.imshow(img,extent=[-2.9,2.98,-1.74,1.53])
        plt.imshow(img,extent=[-1.6,2.45,-1.5,1.35])

    def dumpBoundary(self):
        ip = self.ipBox.value()
        bt = self.btBox.value()
        bp = self.betapBox.value()
        rin = self.rinSlider.value()/10**decimals
        rout = self.routSlider.value()/10**decimals
        k = self.kSlider.value()/10**decimals
        du = self.duSlider.value()/10**decimals
        dl = self.dlSlider.value()/10**decimals
        print('')
        print('Input parameters:')
        print('Ip [m]   Bt [T]   βp       Elon     Up.Tri   Lo.Tri   In.Mid.R Out.Mid.R')
        print(f'{ip:.4f},  {bt:.4f},  {bp:.4f},  {k:.4f},  {du:.4f},  {dl:.4f},  {rin:.4f},  {rout:.4f}')
        print('')
        print('Plasma boundary:')
        print('R [m]   Z [m]')
        for i in range(len(self.rbdry)):
            print(f'{self.rbdry[i]:.4f}, {self.zbdry[i]:.4f}')
        print('')
        if self.plotXptCheckBox.isChecked():
            print('X-points (R, Z):')
            print(f'Lower X-point: {self.rx1:.4f}, {self.zx1:.4f}')
            print(f'Upper X-point: {self.rx2:.4f}, {self.zx2:.4f}')
        
class k2rz():
    def __init__(self,model_path=k2rz_model_path,n_models=10,ntheta=64,closed_surface=True,xpt_correction=True):
        self.nmodels = n_models
        self.ntheta = ntheta
        self.closed_surface = closed_surface
        self.xpt_correction = xpt_correction
        self.models = []
        for i in range(self.nmodels):
            self.models.append(models.load_model(model_path+'/best_model{}'.format(i),custom_objects={'r2_k':self.r2_k}))
    
    def r2_k(self, y_true, y_pred):
        #SS_res = K.sum(K.square(y_true - y_pred))
        #SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        #return ( 1 - SS_res/(SS_tot + epsilon) )
        return 1.0

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


if __name__ == '__main__':
    app = QApplication([])
    window = PBGWidget()
    window.show()
    app.exec()

