#
# Example of how to run the Python code, and access the output
# This case is identical to the default setup of CLASS (the version with interface) 
#

# show the results
from pylab import *
from model import *
import ipywidgets as widgets
from IPython.display import display
from bokeh.io import push_notebook, show, output_notebook, reset_output
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.models import Legend
from collections import OrderedDict

global_handle = None

output_notebook()



class model_interface:
    def __init__(self,inp):
        # define plot:
        self.runlist = OrderedDict()   # define two dictionaries (ordered), one for the dropdownlist, one to link to model input
        self.runlist[inp.name]=inp.name
        self.inplist = OrderedDict()
        self.inplist[inp.name]=inp
        self.inp = inp
# variable used to update of widgets
        self.do_update = True
        # time:
        self.time_l = widgets.Label(value='Time settings')
        self.time_t = widgets.HBox(children=[widgets.FloatText(description='t', value=inp.runtime/3600.),
                                widgets.Label(value='[h]')])
        self.time_dt = widgets.HBox(children=[widgets.IntText(description='\u0394t', value=inp.dt),
                                 widgets.Label(value='[s]')])
        self.time_tdiurnal =  widgets.HBox(children=[widgets.IntText(description='T_diur', value=12),
                                        widgets.Label(value='[h]')])
        self.time_l2 = widgets.Label(value='')
        self.time_l2.layout.height='63pt'
        self.time_v = widgets.VBox(children=[self.time_l,self.time_t,self.time_dt,self.time_tdiurnal,self.time_l2])
        # potential temperature:
        self.pott_l = widgets.Label(value='Potential temperature ')
        self.pott_t0 = widgets.HBox(children=[widgets.IntText(description='\u03b8\u2080', value=inp.theta),
                                widgets.Label(value='[K]')])
        self.pott_dt = widgets.HBox(children=[widgets.IntText(description='\u0394\u03b8\u2080', value=inp.dtheta),
                                widgets.Label(value='[K]')])
        self.pott_gamma = widgets.HBox(children=[widgets.FloatText(description='\u03b3\u03b8', value=inp.gammatheta),
                                widgets.Label(value='[K m\u207B\u00B9]')])
        self.pott_adv = widgets.HBox(children=[widgets.FloatText(description='\u03b8_adv', value=inp.advtheta),
                                widgets.Label(value='[K s\u207B\u00B9]')])
        self.pott_flux = widgets.HBox(children=[widgets.FloatText(description='(w\'\u03b8\')s', value=inp.wtheta),
                                widgets.Label(value='[K m s\u207B\u00B9]')])
        self.pott_diurnal = widgets.Checkbox(value=False,description='Diurnal cycle')
        self.pott_v = widgets.VBox(children=[self.pott_l,self.pott_t0,self.pott_dt,
                                             self.pott_gamma,self.pott_adv,self.pott_flux,
                                             self.pott_diurnal])
        # boundary layer:
        self.bl_l = widgets.Label(value='Boundary Layer')
        self.bl_h0 = widgets.HBox(children=[widgets.IntText(description='h\u2080', value=inp.h),
                                widgets.Label(value='[m]')])
        self.bl_b = widgets.HBox(children=[widgets.FloatText(description='\u03B2', value=inp.beta),
                                 widgets.Label(value='[-]')])
        self.bl_ps =  widgets.HBox(children=[widgets.IntText(description='ps', value=inp.Ps/100.),
                                        widgets.Label(value='[hPa]')])
        self.bl_div =  widgets.HBox(children=[widgets.FloatText(description='div(Uh)', value=inp.divU),
                                        widgets.Label(value='[s\u207B\u00B9]')])
        self.bl_mlffa = widgets.HBox(children=[widgets.Checkbox(value=inp.sw_ml,description='Mixed layer'),
                                   widgets.Checkbox(value=inp.sw_fixft,description='Fix free atmos.')])
        self.bl_v = widgets.VBox(children=[self.bl_l,self.bl_h0,self.bl_b,self.bl_ps,self.bl_div,self.bl_mlffa])
        # moisture:
        self.moist_l = widgets.Label(value='Moisture')
        self.moist_q0 = widgets.HBox(children=[widgets.FloatText(description='q\u2080', value=inp.q*1e3),
                                widgets.Label(value='[g kg\u207B\u00B9]')])
        self.moist_dq = widgets.HBox(children=[widgets.FloatText(description='\u0394q\u2080', value=inp.dq*1e3),
                                widgets.Label(value='[g kg\u207B\u00B9]')])

        self.moist_gamma = widgets.HBox(children=[widgets.FloatText(description='\u03B3_q', value=inp.gammaq*1e3),
                                widgets.Label(value='[g kg\u207B\u00B9 m\u207B\u00B9]')])

        self.moist_adv = widgets.HBox(children=[widgets.FloatText(description='q_adv ', value=inp.advq*1e3),
                                widgets.Label(value='[g kg\u207B\u00B9 s\u207B\u00B9]')])
        self.moist_flux = widgets.HBox(children=[widgets.FloatText(description='(w\'q\')s', value=inp.wq*1e3),
                                widgets.Label(value='[g kg\u207B\u00B9 m s\u207B\u00B9]')])
        self.moist_diurnal = widgets.Checkbox(value=False,description='Diurnal cycle')

        self.moist_v =  widgets.VBox(children=[self.moist_l,self.moist_q0,self.moist_dq,
                                          self.moist_gamma,self.moist_adv,self.moist_flux,
                                          self.moist_diurnal])

        self.left_c = widgets.VBox(children=[self.time_v,self.pott_v])


        self.right_c = widgets.VBox(children=[self.bl_v,self.moist_v])
        
        self.run_name = widgets.Text(value = inp.name, description='Name')
#        self.run_list = widgets.Dropdown( options=self.runlist)
        self.run_list = widgets.Dropdown( options=self.runlist, value=inp.name)

        self.basic = widgets.HBox(children=[self.left_c,self.right_c])

        self.children = [self.basic,self.time_v,self.time_v,self.time_v]
        self.tab = widgets.Tab(children=self.children,_titles= 
               {0: 'Basic', 1: 'Wind',2: 'Rad / Geo', 3: 'Surface'})
        self.option_run = widgets.Button(description='Run/Plot')
        self.option_new = widgets.Button(description='New')
        self.option_clone = widgets.Button(description='Clone')
        self.option_delete = widgets.Button(description='Delete')
        self.option_run.layout.width='80pt'
        self.option_new.layout.width='80pt'
        self.option_clone.layout.width='80pt'
        self.option_delete.layout.width='80pt'
        self.run_name.layout.width='150pt'
        self.run_list.layout.width='150pt'
        self.bottom = widgets.HBox(children = [self.option_run,self.option_new,self.option_clone,
                                               self.option_delete,self.run_name,self.run_list])
        # which widgets are observed?
        self.option_run.on_click(self.classmodel)
        self.option_new.on_click(self.newrun)
        self.option_clone.on_click(self.clonerun)
        self.option_delete.on_click(self.deleterun)
        #self.run_name.on_submit(self.name_changed)   # name has been typed
        self.run_name.observe(self.name_changed, names='value')
        self.run_list.observe(self.list_changed, names='value')   # something in runlist changed!
        # plot window:
        self.xaxis = widgets.SelectMultiple(options=['-'],value=('-',),description = 'x-axis')
        self.yaxis = widgets.SelectMultiple(options=['-'],value=('-',),description = 'y-axis')
        self.plot_selection = widgets.SelectMultiple(options=['-'],value=('-',),description = 'runs',visible=False)
        self.xaxis.layout.width='200pt'
        self.yaxis.layout.width='200pt'
        self.plot_selection.layout.width='200pt'
        self.xaxis.observe(self.class_plot, names='value')   # something in runlist changed!
        self.yaxis.observe(self.class_plot, names='value')   # something in runlist changed!
        self.axis = widgets.HBox(children = [self.xaxis,self.yaxis,self.plot_selection])

#        self.p = figure(title="Class Output", plot_height=300, plot_width=600, y_range=(0,2000),x_range=(0,24))
        #prepare for multiple run outputs in the same plot:
#        x = [0,0]
#        y = [0,0]
        self.colors = ['pink','orange','blue','green','black','cyan','red']
#        for i in range(5):
#            self.r.append(self.p.line( x, y, color=self.colors[i], line_width=3))
        display(self.tab)
        display(self.bottom)
        display(self.axis)
        self.do_plot = widgets.ToggleButton(description='Refresh Plots')
        widgets.interact(self.doplot, do_plot=self.do_plot)
        

        

    def set_state(self):
        ''' set the state of the tab widget into the run defined in the run_name field'''
        name = self.run_name.value
        inp = self.inplist.get(name)
        self.read_input(inp)
        self.runlist[name]=name
        self.inplist[name]=inp
    
    def printstate(self):
        for inp in self.inplist:
            xin = self.inplist.get(inp)
            print(inp,xin.wtheta)
        
    def get_state(self):
        ''' get the state of the tab widget from the run defined in the run_name field'''
        name = self.run_name.value
        inp = self.inplist.get(name)   # get inp from Dict
        self.set_input(inp)
        
    def newrun(self,btt):
        # first set the current state:
        self.set_state()
        name = self.run_name.value[:-1]+str(len(self.run_list.options)+1)
        x1 = model_instance(name)
        
        # Documentation for this trick: https://github.com/jupyter-widgets/ipywidgets/issues/1762
        run_options_copy = self.runlist.copy()
        run_options_copy[name] = name
        self.run_list.options = run_options_copy
             
        self.inplist[name] = x1.runinput
        # set widgets to these value
        self.run_list.value = name
        self.set_input(x1.runinput)
        self.run_name.value = name
        self.do_update = True

       
    def clonerun(self,btt):
        from copy import deepcopy
        self.set_state()
        # number selected
        name = self.run_list.value
        # set new name:
        name_clone = name+' (clone)'
        newinplist = deepcopy(self.inplist.get(name))
        newinplist.name = name_clone
        
        # Documentation for this trick: https://github.com/jupyter-widgets/ipywidgets/issues/1762
        run_options_copy = self.runlist.copy()
        run_options_copy[name_clone] = name_clone
        self.run_list.options = run_options_copy

        self.inplist[name_clone]=newinplist
        # select the last newly created runinstance
        # This seems to crash
        self.run_list.value = name_clone
        self.set_input(newinplist)
        self.run_name.value = name_clone
        self.do_update = True
       
    def deleterun(self,btt):
        selected = self.run_list.value
        options = self.run_list.options
        if len(options)>1:
            self.runlist.pop(selected)
            self.inplist.pop(selected)
            # This seems not to work 
            # options.remove(selected)
            del options[selected]
            self.run_list.options = options
            # select the last value from the list
            self.set_input(self.inplist.get(options[0]))
            self.run_name.value = options[0]
            self.run_list.value = options[0]
        
    def name_changed(self,sender):
        new_name = self.run_name.value
        options = list(self.run_list.options)
        #print('name_changed',new_name)
        #avoid empty and exisitng names:
        if new_name != '' and options.count(new_name) == 0:
            #  update the name in the run_list:
            selected = self.run_list.value
            #print('name_chnaged:',options)
#     note that name change is also called after delete, so this might fail:
            isel = options.count(selected)
            if isel > 0:
                options[options.index(selected)] = new_name
                #print('name_chnaged:',options)
# also change the dictionaries such that order is maintained:
                newr = OrderedDict()
                newi = OrderedDict()
                for key in self.runlist:
                    if key == selected:
                        newr[new_name] = new_name
                        newi[new_name] = self.inplist.get(key)
                    else:
                        newr[key]=key
                        newi[key]=self.inplist.get(key)
                self.runlist=newr
                self.inplist=newi
                self.do_update = False
                self.run_list.options = options
                self.do_update = False
                self.run_list.value = new_name
        self.do_update = True
            
    def list_changed(self,change):
        self.set_state()
        if self.do_update:
            selected = self.run_list.value 
            self.run_name.value = selected
            self.get_state()
        self.do_update = True

    def doplot(self, do_plot=True):
        self.classmodel(do_plot)
        self.do_plot.value = False

    def classmodel(self,change):
        # things might have been set: set the state:
#        self.reset_plot()
        self.set_state()
        self.plot_selection.visible=True
        self.plot_selection.options=list(self.run_list.options)
        # set and run all models
        self.plot_selection.value=list(self.run_list.options)
        runlist = self.plot_selection.value
        self.outp = [] 
        for runname in runlist:
            inp = self.inplist.get(runname)
            inp.name = runname
            r1 = model(inp)
            r1.run()
            self.outp.append(r1)
        
        options = ['t','h','theta','q']
        self.yaxis.options = options
        self.xaxis.options = options
        # set empty tuple to force plot on widget change:
        self.yaxis.value = ()
        self.xaxis.value = ()
#            axp = ax[0]
#            axp.plot(r1.out.t, r1.out.h,label=runname)
#            axp = ax[1]
#            axp.plot(r1.out.t, r1.out.theta,label=runname)
#            axp = ax[2]
#            axp.plot(r1.out.t, r1.out.q*1000.,label=runname)
#        for axp in ax:
#            axp.grid(True)
#            axp.legend(loc='best')
#        ax[0].set_ylabel('h [m]')
#        ax[1].set_ylabel('theta [K]')
#        ax[2].set_ylabel('q [g kg-1]')
#        ax[2].set_xlabel('time [h]')

    def class_plot(self,change):
        global global_handle
        nx = len(self.xaxis.value)
        ny = len(self.yaxis.value)
        selected = self.plot_selection.value
        if nx*ny > 0:
            reset_output()
            output_notebook()
            
            p_all = []
            for xxa in self.xaxis.value:
                for iax,yya in enumerate(self.yaxis.value):
                    p=figure(title="Class Output", plot_height=300)
                    xmax = -1e9
                    xmin = 1e9
                    ymax = -1e9
                    ymin = 1e9
                    legend_items = []
                    r = []
                    for ir,irun in enumerate(self.outp):
                        name = irun.input.name
                        if selected.count(name) == 1:
                            if xxa == 't':
                                x = irun.out.t
                                xlab = 'time [h]' 
                            elif xxa == 'h':
                                x = irun.out.h
                                xlab = 'h [m]'
                            elif xxa == 'theta':
                                x = irun.out.theta
                                xlab = 'theta [K]'
                            elif xxa == 'q':
                                x = irun.out.q*1000
                                xlab = 'g [g/kg]'
                            else:
                                x = None
                                xlab = ''
                            xmax = max(xmax,max(x))
                            xmin = min(xmin,min(x))
                            
                            if yya == 't':
                                y = irun.out.t
                                ylab = 'time [h]' 
                            elif yya == 'h':
                                y = irun.out.h
                                ylab = 'h [m]'
                            elif yya == 'theta':
                                y = irun.out.theta
                                ylab = 'theta [K]'
                            elif yya == 'q':
                                y = irun.out.q*1000
                                ylab = 'g [g/kg]'
                            else:
                                y = None
                                ylab = ''
                            ymax = max(ymax,max(y))
                            ymin = min(ymin,min(y))
                            #self.r[ir].data_source.data['x'] = x
                            #self.r[ir].data_source.data['y'] = y
                            r.append(p.line( x, y, color=self.colors[ir], legend=name, line_width=3))
                            #legend_items.append((name,r[ir]))
                    p.x_range.start=xmin
                    p.x_range.end  =xmax
                    p.y_range.start=ymin
                    p.y_range.end  =ymax
                    p.xaxis.axis_label = xlab
                    p.yaxis.axis_label = ylab
                    p.legend.location = "top_left"
                    p_all.append(p)
                # show(column(p_all))
                # does not work yet to keep one graph. Idea comes from https://bokeh.pydata.org/en/latest/docs/user_guide/notebook.html
                # Or this may work: http://localhost:8889/notebooks/Untitled3.ipynb?kernel_name=python3
                if (not global_handle):
                    global_handle = show(obj=p, new='tab',notebook_handle=True)
                else:
                    push_notebook(handle=global_handle)
                    #legend = Legend(items=legend_items,location=(0, -30))
                    #p.add_layout(legend, 'center')




#                        axp.plot(x,y,label=irun.input.name)
#                    axp.grid(True)
#                    axp.set_ylabel(ylab)
#                    axp.legend(loc='best')
#                axp.set_xlabel(xlab)
#                push_notebook()


    def read_input(self,inp):
        ''' convert the status if the widget input to valid model input. Input should be an instance of model input '''
        # time:
        inp.runtime     = self.time_t.children[0].value*3600.   # [s]
        inp.dt          = self.time_dt.children[0].value*1.0    # [s]
        # potential temperature:
        inp.theta       = self.pott_t0.children[0].value*1.0    # [K]
        inp.dtheta      = self.pott_dt.children[0].value*1.0    # [K]
        inp.gammatheta  = self.pott_gamma.children[0].value     # [K/m]
        inp.advtheta    = self.pott_adv.children[0].value       # [K/s] 
        inp.wtheta      = self.pott_flux.children[0].value      # [K m/s]
        # boundary layer:
        inp.h           =  self.bl_h0.children[0].value*1.0     # [m]
        inp.beta        =  self.bl_b.children[0].value          # [-]
        inp.Ps          =  self.bl_ps.children[0].value*100.0   # [Pa]
        inp.divU        =  self.bl_div.children[0].value        # [s-1]
        inp.sw_ml       =  self.bl_mlffa.children[0].value      # [True/False]
        inp.sw_fixft    =  self.bl_mlffa.children[1].value      # [True/False]
        # moisture:
        inp.q           =  self.moist_q0.children[0].value*1e-3       #  kg/kg
        inp.dq          =  self.moist_dq.children[0].value*1e-3       #  kg/kg
        inp.gammaq      =  self.moist_gamma.children[0].value*1e-3    #  kg/kg /m
        inp.advq        =  self.moist_adv.children[0].value*1e-3      #  kg/kg /s
        inp.wq          =  self.moist_flux.children[0].value*1e-3     #  kg/kg m/s
        
    def set_input(self,inp):
        ''' convert the status input to widget'''
        # time:
        self.time_t.children[0].value = inp.runtime/3600.   # [s]
        self.time_dt.children[0].value = inp.dt
        # potential temperature:
        self.pott_t0.children[0].value = inp.theta
        self.pott_dt.children[0].value = inp.dtheta
        self.pott_gamma.children[0].value = inp.gammatheta
        self.pott_adv.children[0].value = inp.advtheta
        self.pott_flux.children[0].value = inp.wtheta
        # boundary layer:
        self.bl_h0.children[0].value = inp.h
        self.bl_b.children[0].value = inp.beta
        self.bl_ps.children[0].value = inp.Ps/100.0  
        self.bl_div.children[0].value = inp.divU 
        self.bl_mlffa.children[0].value = inp.sw_ml
        self.bl_mlffa.children[1].value = inp.sw_fixft
        # moisture:
        self.moist_q0.children[0].value = inp.q*1e3
        self.moist_dq.children[0].value = inp.dq*1e3
        self.moist_gamma.children[0].value = inp.gammaq*1e3
        self.moist_adv.children[0].value = inp.advq*1e3
        self.moist_flux.children[0].value = inp.wq*1e3

        
         
class model_instance:
    def __init__(self,name):
        """ 
        Create empty model_input and set up case
        """
        run1input = model_input()
        
        run1input.name = name

        run1input.dt         = 60.       # time step [s]
        run1input.runtime    = 12*3600    # total run time [s]

        # mixed-layer input
        run1input.sw_ml      = True      # mixed-layer model switch
        run1input.sw_shearwe = False     # shear growth mixed-layer switch
        run1input.sw_fixft   = False     # Fix the free-troposphere switch
        run1input.h          = 200.      # initial ABL height [m]
        run1input.Ps         = 101300.   # surface pressure [Pa]
        run1input.divU       = 0.        # horizontal large-scale divergence of wind [s-1]
        run1input.fc         = 1.e-4     # Coriolis parameter [m s-1]

        run1input.theta      = 288.      # initial mixed-layer potential temperature [K]
        run1input.dtheta     = 1.        # initial temperature jump at h [K]
        run1input.gammatheta = 0.006     # free atmosphere potential temperature lapse rate [K m-1]
        run1input.advtheta   = 0.        # advection of heat [K s-1]
        run1input.beta       = 0.2       # entrainment ratio for virtual heat [-]
        run1input.wtheta     = 0.1       # surface kinematic heat flux [K m s-1]

        run1input.q          = 0.008     # initial mixed-layer specific humidity [kg kg-1]
        run1input.dq         = -0.001    # initial specific humidity jump at h [kg kg-1]
        run1input.gammaq     = 0.        # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
        run1input.advq       = 0.        # advection of moisture [kg kg-1 s-1]
        run1input.wq         = 0.1e-3    # surface kinematic moisture flux [kg kg-1 m s-1]

        run1input.CO2        = 422.      # initial mixed-layer CO2 [ppm]
        run1input.dCO2       = -44.      # initial CO2 jump at h [ppm]
        run1input.gammaCO2   = 0.        # free atmosphere CO2 lapse rate [ppm m-1]
        run1input.advCO2     = 0.        # advection of CO2 [ppm s-1]
        run1input.wCO2       = 0.        # surface kinematic CO2 flux [ppm m s-1]

        run1input.sw_wind    = False     # prognostic wind switch
        run1input.u          = 6.        # initial mixed-layer u-wind speed [m s-1]
        run1input.du         = 4.        # initial u-wind jump at h [m s-1]
        run1input.gammau     = 0.        # free atmosphere u-wind speed lapse rate [s-1]
        run1input.advu       = 0.        # advection of u-wind [m s-2]

        run1input.v          = -4.0      # initial mixed-layer u-wind speed [m s-1]
        run1input.dv         = 4.0       # initial u-wind jump at h [m s-1]
        run1input.gammav     = 0.        # free atmosphere v-wind speed lapse rate [s-1]
        run1input.advv       = 0.        # advection of v-wind [m s-2]

        run1input.sw_sl      = False     # surface layer switch
        run1input.ustar      = 0.3       # surface friction velocity [m s-1]
        run1input.z0m        = 0.02      # roughness length for momentum [m]
        run1input.z0h        = 0.002     # roughness length for scalars [m]

        run1input.sw_rad     = False     # radiation switch
        run1input.lat        = 51.97     # latitude [deg]
        run1input.lon        = -4.93     # longitude [deg]
        run1input.doy        = 268.      # day of the year [-]
        run1input.tstart     = 6.8       # time of the day [h UTC]
        run1input.cc         = 0.0       # cloud cover fraction [-]
        run1input.Q          = 400.      # net radiation [W m-2] 
        run1input.dFz        = 0.        # cloud top radiative divergence [W m-2] 

        run1input.sw_ls      = False     # land surface switch
        run1input.ls_type    = 'js'      # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
        run1input.wg         = 0.21      # volumetric water content top soil layer [m3 m-3]
        run1input.w2         = 0.21      # volumetric water content deeper soil layer [m3 m-3]
        run1input.cveg       = 0.85      # vegetation fraction [-]
        run1input.Tsoil      = 285.      # temperature top soil layer [K]
        run1input.T2         = 286.      # temperature deeper soil layer [K]
        run1input.a          = 0.219     # Clapp and Hornberger retention curve parameter a
        run1input.b          = 4.90      # Clapp and Hornberger retention curve parameter b
        run1input.p          = 4.        # Clapp and Hornberger retention curve parameter c
        run1input.CGsat      = 3.56e-6   # saturated soil conductivity for heat

        run1input.wsat       = 0.472     # saturated volumetric water content ECMWF config [-]
        run1input.wfc        = 0.323     # volumetric water content field capacity [-]
        run1input.wwilt      = 0.171     # volumetric water content wilting point [-]

        run1input.C1sat      = 0.132     
        run1input.C2ref      = 1.8

        run1input.LAI        = 2.        # leaf area index [-]
        run1input.gD         = 0.0       # correction factor transpiration for VPD [-]
        run1input.rsmin      = 110.      # minimum resistance transpiration [s m-1]
        run1input.rssoilmin  = 50.       # minimun resistance soil evaporation [s m-1]
        run1input.alpha      = 0.25      # surface albedo [-]

        run1input.Ts         = 290.      # initial surface temperature [K]

        run1input.Wmax       = 0.0002    # thickness of water layer on wet vegetation [m]
        run1input.Wl         = 0.0000    # equivalent water layer depth for wet vegetation [m]

        run1input.Lambda     = 5.9       # thermal diffusivity skin layer [-]

        run1input.c3c4       = 'c3'      # Plant type ('c3' or 'c4')

        run1input.sw_cu      = False     # Cumulus parameterization switch
        run1input.dz_h       = 150.      # Transition layer thickness [m]
        self.runinput = run1input


