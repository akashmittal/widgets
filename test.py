import ipywidgets as widgets
import pandas as pd
from IPython.display import display, clear_output
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA


class test_class(object):
    
    def __init__(self):
        tab_contents = ['Data Load', 'Data Extraction and Visualization', 'Data Model', 'Results']

        tab = widgets.Tab()
        
        for i, title in enumerate(tab_contents):#    range(len(tab_contents)):
            tab.set_title(i, title)

        children = [widgets.Text(description=name) for name in tab_contents]
        
        self.data_load_children()
        
        self.data_analyze_children()
        
        self.data_model_children()
        
        tab.children = [self.data_load_children , self.data_analyze_children, self.data_model_children]
            
        display(tab)
        
    def data_load_children(self):
        
        widget_file_name = widgets.Text(description = 'File name', value = 'Airpassengers.csv')
        
        widget_out = widgets.Output()
        
        widget_button_load = widgets.Button(
                                    description='Load data',
                                    disabled=False,
                                    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                    tooltip='Load data',
                                    icon='check'
                                )
        
        def run(v):
            self.data = pd.read_csv(widget_file_name.value)
            widget_button_load.button_style = 'success'
            with widget_out:
                clear_output()
                con=self.data['Month']
                self.data['Month']=pd.to_datetime(self.data['Month'])
                self.data.set_index('Month', inplace=True)
                self.ts = self.data['#Passengers']
                print(self.ts.head())
        
        widget_button_load.on_click(run)
        
        self.data_load_children = widgets.VBox([widget_file_name, widget_button_load, widget_out])

    def data_analyze_children(self):
        
        widget_button_analyze = widgets.Button(
                                    description='Analzye data',
                                    disabled=False,
                                    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                    tooltip='Analyze data',
                                    icon='check'
                                )
        
        widget_out = widgets.Output()
        
        def analyze(v):
            with widget_out:
                clear_output()
                plt.figure()
                plt.plot(self.ts)
                plt.title('Original Data')
                plt.ylabel('Number of passengers')
                plt.xlabel('Time- Original')
                
                print ('Check for the Stationarity of the series using Dickey-Fuller Test:')
                dftest = adfuller(self.ts, autolag='AIC')
                dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
                for key,value in dftest[4].items():
                    dfoutput['Critical Value (%s)'%key] = value
                print (dfoutput)
                
                print('\nTransformation required to remove the trend in variance')
                
                widget_options = widgets.Dropdown(
                    options=['none','log', 'sqrt', 'cube-root'],
                    value='none',
                    description='transformation type:',
                    disabled=False,
                )
                
                display(widget_options)
                
                widget_button_transform = widgets.Button(
                                    description='Transform data and remove trend',
                                    disabled=False,
                                    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                    tooltip='Transform data and remove trend',
                                    icon='check'
                                )
                
                
                wid_transform_out = widgets.Output()
                
                display(widgets.VBox([widget_button_transform, wid_transform_out]))

                def transform(v):
                    with wid_transform_out:
                        clear_output()
                        if widget_options.value == 'none':
                            self.ts_trans = self.ts
                        elif widget_options.value == 'log':
                            self.ts_trans = np.log(self.ts)
                        elif widget_options.value == 'sqrt':
                            self.ts_trans = np.sqrt(self.ts)
                        else :
                            self.ts_trans = self.ts

                        plt.figure()
                        #plt.plot(self.ts, label = 'Original')
                        plt.plot(self.ts_trans, label = 'Transformed')
                        plt.legend()
                        
                        
                        moving_avg = self.ts_trans.rolling(window=12,center=False).mean()
                        ts_trans_ma_diff = self.ts_trans - moving_avg
                        ts_trans_ma_diff.dropna(inplace=True)
                        
                        self.ts_diff = self.ts_trans - self.ts_trans.shift()
                        
                        plt.plot(self.ts_diff, label = 'Transformed and diffed')
                        plt.legend()
                        
                        
                        
                
                widget_button_transform.on_click(transform)

        widget_button_analyze.on_click(analyze)
                
        self.data_analyze_children = widgets.VBox([widget_button_analyze, widget_out])
        
        
    def data_model_children(self):
        widget_button_acf= widgets.Button(
                                    description='Check the Autocorrelation lags',
                                    disabled=False,
                                    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                    tooltip='Check the Autocorrelation lags',
                                    icon='check'
                                )
        
        widget_out_acf = widgets.Output()
        
        widget_button_arima= widgets.Button(
                                    description='ARIMA Model',
                                    disabled=True,
                                    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                                    tooltip='ARIMA Model',
                                    icon='check'
                                )
        p = widgets.IntText(value=2, description = 'P')
        d = widgets.IntText(value=1, description = 'D')
        q = widgets.IntText(value=0, description = 'Q')
        
        
        
        def acf_run(v):
            ts = self.ts_diff.dropna()
            with widget_out_acf:
                
                clear_output()
                
                
                lag_acf = acf(ts , nlags=20)
                lag_pacf = pacf(ts, nlags=20, method = 'ols')
                
                # PLOT ACF:
                plt.figure()
                plt.subplot(121)
                plt.plot(lag_acf)
                plt.axhline(y=0, linestyle = '--', color = 'gray')
                plt.axhline(y=-1.96/np.sqrt(len(self.ts_diff)), linestyle = '--', color = 'gray')
                plt.axhline(y=1.96/np.sqrt(len(self.ts_diff)), linestyle = '--', color = 'gray')
                plt.title('Autocorrelation function')
        
                # PLOT PACF:
                plt.subplot(122)
                plt.plot(lag_pacf)
                plt.axhline(y=0, linestyle = '--', color = 'gray')
                plt.axhline(y=-1.96/np.sqrt(len(self.ts_diff)), linestyle = '--', color = 'gray')
                plt.axhline(y=1.96/np.sqrt(len(self.ts_diff)), linestyle = '--', color = 'gray')
                plt.title('Partial Autocorrelation function')
                plt.tight_layout()
                widget_button_arima.disabled = False
                
                display(widgets.HBox([p,d,q]))
                
                        
        widget_button_acf.on_click(acf_run)

        out_arima = widgets.Output()
        
        def arima_run(v):
            with out_arima:
                clear_output()
                model = ARIMA(self.ts_trans, order = (p.value, d.value, q.value))
                self.results_AR = model.fit(disp=1)
                plt.figure()
                plt.plot(self.ts_diff)
                plt.plot(self.results_AR.fittedvalues, color='red')
                plt.title('RSS : %.4f' % np.nansum((self.results_AR.fittedvalues - self.ts_diff)**2))
        
        widget_button_arima.on_click(arima_run) 
                
        self.data_model_children = widgets.VBox([widget_button_acf, widget_out_acf, widget_button_arima, out_arima])
        
        
    def results_child(Self):
        
        pass
        
        
        