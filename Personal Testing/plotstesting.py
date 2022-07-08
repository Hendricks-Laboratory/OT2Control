#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:14:58 2022

@author: grantdidway
"""

def main():
    plot = Plotter('CNH_test')

class Plotter():
    '''
    This class creates and saves plots 
    
    ATTRIBUTES:  
        df df: Not passed in but created in init. Pandas Dataframe to be used 
            to store all scans from the run.
        str data_path: pathname for local platereader data.
    
    METHODS:  
        add_to_df() void: formats data from a scan and adds to the data frame.
    '''
    
    def __init__(self, filename):
        self.filename = filename
    
    def _create_plot(self, row, i):
        '''
        exectues a plot command  
        params:  
            pd.Series row: a row of self.rxn_df  
            int i: index of this row  
        '''
        wellnames = row[self._products][row[self._products].astype(bool)].index
        plot_type = row['plot_protocol']
        filename = row['plot_filename']
        #make sure you have mapping for all files

        self._update_cached_locs(wellnames)
        pr_dict = {self._cached_reader_locs[wellname].loc: wellname for wellname in wellnames}
        #it's not safe to plot in simulation because the scan file may not exist yet
        df, metadata = self.pr.load_reader_data(row['scan_filename'], pr_dict)
        #execute the plot depending on what was specified
        if plot_type == 'single_kin':
            for wellname in wellnames:
                self.plot_single_kin(df, metadata['n_cycles'], wellname, "{}_{}".format(wellname, filename))
        elif plot_type == 'overlay':
            self.plot_LAM_overlay(df, wellnames, filename)
        elif plot_type == 'multi_kin':
            self.plot_kin_subplots(df, metadata['n_cycles'], wellnames, filename)
            
    def _plot_setup_overlay(self,title):
        '''
        Sets up a figure for an overlay plot  
        params:  
            str title: the title of the reaction  
        '''
        #formats the figure nicely
        plt.figure(num=None, figsize=(4, 4),dpi=300, facecolor='w', edgecolor='k')
        plt.legend(loc="upper right",frameon = False, prop={"size":7},labelspacing = 0.5)
        plt.rc('axes', linewidth = 2)
        plt.xlabel('Wavelength (nm)',fontsize = 16)
        plt.ylabel('Absorbance (a.u.)', fontsize = 16)
        plt.tick_params(axis = "both", width = 2)
        plt.tick_params(axis = "both", width = 2)
        plt.xticks([300,400,500,600,700,800,900,1000])
        plt.yticks([i/10 for i in range(0,11,1)])
        plt.axis([300, 1000, 0.0 , 1.0])
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.title(str(title), fontsize = 16, pad = 20)
        
    def plot_LAM_overlay(self,df,wells,filename=None):
        '''
        plots overlayed spectra of wells in the order that they are specified  
        params:  
            df df: dataframe with columns = chem_names, and values of each column is a series
              of scans in 701 intervals.  
            str filename: the title of the plot, and the file  
            list<str> wells: an ordered list of all of the chem_names you want to plot.  
        Postconditions:  
            plot has been written with name "overlay.png" to the plotting dir. or 
            {filename}.png if filename was supplied  
        '''
        if not filename:
            filename = "overlay"
        x_vals = list(range(300,1001))
        #overlays only things you specify
        y = []
        #df = df[df_reorder]
        #headers = [well_key[k] for k in df.columns]
        #legend_colors = []
        for chem_name in wells:
            y.append(df[chem_name].iloc[-701:].to_list())
        self._plot_setup_overlay(filename)
        colors = list(cm.rainbow(np.linspace(0, 1,len(y))))
        for i in range(len(y)):
            plt.plot(x_vals,y[i],color = tuple(colors[i]))
        patches = [mpatches.Patch(color=color, label=label) for label, color in zip(wells, colors)]
        plt.legend(patches, wells, loc='upper right', frameon=False,prop={'size':3})
        legend = pd.DataFrame({'Color':patches,'Labels': wells})
        plt.savefig(os.path.join(self.plot_path, '{}.png'.format(filename)))
        plt.close()
       
    # below until ~end is all not used yet needs to be worked up
    def plot_kin_subplots(self,df,n_cycles,wells,filename=None):
        '''
        TODO this function doesn't save properly, but it does show. Don't know issue  
        plots kinetics for each well in the order given by wells.  
        params:  
            df df: the scan data  
            int n_cycles: the number of cycles for the scan data  
            list<str> wells: the wells you want to plot in order
        Postconditions:  
            plot has been written with name "{filename}_overlay.png" to the plotting dir.  
            If filename is not supplied, name is kin_subplots
        '''
        if not filename:
            filename='kin_subplots'
        x_vals = list(range(300,1001))
        colors = list(cm.rainbow(np.linspace(0, 1, n_cycles)))
        fig, axes = plt.subplots(8, 12, dpi=300, figsize=(50, 50),subplot_kw=dict(box_aspect=1,sharex = True,sharey = True))
        for idx, (chem_name, ax) in enumerate(zip(wells, axes.flatten())):
            ax.set_title(chem_name)
            self._plot_kin(ax, df, n_cycles, chem_name)
            plt.subplots_adjust(wspace=0.3, hspace= -0.1)
        
            ax.tick_params(
                which='both',
                bottom='off',
                left='off',
                right='off',
                top='off'
            )
            ax.set_xlim((300,1000))
            ax.set_ylim((0,1.0))
            ax.set_xlabel("Wavlength (nm)")
            ax.set_ylabel("Absorbance (A.U.)")
            ax.set_xticks(range(301, 1100, 100))
            #ax.set_aspect(adjustable='box')
            #ax.set_yticks(range(0,1))
        else:
            [ax.set_visible(False) for ax in axes.flatten()[idx+1:]]
        plt.savefig(os.path.join(self.plot_path, '{}.png'.format(filename)))
        plt.close()

    def _plot_kin(self, ax, df, n_cycles, chem_name):
        '''
        helper method for kinetics plotting methods  
        params:  
            plt.axes ax: or anything with a plot func. the place you want ot plot  
            df df: the scan data  
            int n_cycles: the number of cycles in per well scanned  
            str chem_name: the name of the chemical to be plotted  
        Postconditions:  
            a kinetics plot of the well has been plotted on ax  
        '''
        x_vals = list(range(300,1001))
        colors = list(cm.rainbow(np.linspace(0, 1, n_cycles)))
        kin = 0
        col = df[chem_name]
        for kin in range(n_cycles):
            ax.plot(x_vals, df[chem_name].iloc[kin*701:(kin+1)*701],color=tuple(colors[kin]))
        
    
    def plot_single_kin(self, df, n_cycles, chem_name, filename=None):
        '''
        plots one kinetics trace. 
        params:  
            df df: the scan data  
            int n_cycles: the number of cycles in per well scanned  
            str chem_name: the name of the chemical to be plotted  
            str filename: the name of the file to write  
        Postconditions:  
            A kinetics trace of the well has been written to the Plots directory.
            under the name filename. If filename was None, the filename will be 
            {chem_name}_kinetics.png
        '''
        if not filename:
            filename = '{}_kinetics'.format(chem_name)
        self._plot_setup_overlay('Kinetics {}: '.format(chem_name))
        self._plot_kin(plt,df, n_cycles, chem_name)
        plt.savefig(os.path.join(self.plot_path, '{}.png'.format(filename)))
        plt.close()
    
if __name__ == '__main__':
    main()