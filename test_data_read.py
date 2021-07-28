import controller
from controller import PlateReader
import socket

rxn_sheet_name = 'test_plotting'
use_cache=True
my_ip = socket.gethostbyname(socket.gethostname())
controller = controller.ProtocolExecutor(rxn_sheet_name, my_ip, '', use_cache=use_cache)

controller._init_pr(simulate=True)
params = [
('MSR_034_210706.csv',{'A07':'W1', 'B07':'W2', 'C07':'W3', 'D07':'W4', 'E07':'W5', 'F07':'W6', 'G07':'W7', 'H07':'W8'},'plot1', ['W1','W2','W3','W4','W5','W6','W7','W8'])

]

try:
    for param in params:
        df, metadata = controller.pr.load_data(*param[0:2])
        controller.plot_LAM_overlay(df,param[2],param[3])
except Exception as e:
    controller.pr.shutdown()
    raise e
controller.pr.shutdown()
