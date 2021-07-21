from ot2lib import PlateReader

pr = PlateReader(0)
#pr.edit_layout('NC synthesis kinetics (60s stir)',well_list)
pr.run_protocol('NC synthesis kinetics (60s stir)','scan_all_wells', layout='all')
#print('shaking')
#pr.shake()
#print('shook')
#pr.run_protocol('NC synthesis kinetics (60s stir)', 'Test_results_Noah_All_20Jul21', layout=['G1'])
pr.shutdown()
