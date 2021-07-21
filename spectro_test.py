from ot2lib import PlateReader

def print_file():
    with open('/mnt/c/Program Files/SPECTROstar Nano V5.50/SPECTROstar Nano.ini','r') as config:
        print(config.read())
print_file()
pr = PlateReader(1)
print_file()
pr.exec_macro('PlateOut')
#print('shaking')
#pr.shake()
#print('shook')
#pr.run_protocol('NC synthesis kinetics (60s stir)', 'Test_results_Noah_All_20Jul21', layout=['G1'])
pr.shutdown()
print_file()
