from spectro_interface import PlateReader

pr = PlateReader()
#pr.shake_protocol()
print("running  synthesis kinetics (60s stir)")
pr.run_protocol('NC synthesis kinetics (60s stir)', layout=['A1','A2','A3'])
print("shutting down")
pr.shutdown()
