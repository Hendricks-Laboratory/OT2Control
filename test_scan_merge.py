from controller import DummyReader

pr = DummyReader('Controller_Out/Test_Callbacks/pr_data')

pr.merge_scans(['scan1-a.csv'], 'scan1.csv')
pr.merge_scans(['scan2-a.csv', 'scan2-b.csv'], 'scan2.csv')
