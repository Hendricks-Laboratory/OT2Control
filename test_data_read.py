from controller import PlateReader
data_path = 'Data_Analysis'
params = [
('MSR_034_210706.csv',{'A07':'W1', 'B07':'W2', 'C07':'W3', 'D07':'W4', 'E07':'W5', 'F07':'W6', 'G07':'W7', 'H07':'W8'}),

('test_inLab02Results.csv',{'B05':'W1', 'C05':'W2', 'D05':'W3', 'E05':'W4'})
]

try:
    pr = PlateReader(simulate=True)

    for param in params:
        df = pr.load_reader_data(*param)
        print(df)
except Exception as e:
    pr.shutdown()
    raise e
pr.shutdown()
