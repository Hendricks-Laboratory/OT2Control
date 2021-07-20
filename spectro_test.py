from spectro_interface import PlateReader

pr = PlateReader()
#pr.shake_protocol()
pr.run_protocol('NC synthesis kinetics (60s stir)', 'Test_results_Noah_All_20Jul21', layout=['G1'])
pr.shutdown()
