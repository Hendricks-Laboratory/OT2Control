from tensorboard import program
import time
# logdir0= "logs0/fit/" 

# logdir2= "logs2/fit/" 

# logdirNew= "logsNew/fit/" 

logdirNew_2= "logsNew_2/fit/" 

# logdir11= "logs11/fit/" 

# logdir11_2= "logs11_2/fit/" 




# tb = program.TensorBoard()
# tb1= program.TensorBoard()
# tb2= program.TensorBoard()
tb3= program.TensorBoard()
# tb4= program.TensorBoard()
# tb5= program.TensorBoard()

# tb.configure(argv=[None, '--logdir', logdir0])
# tb1.configure(argv=[None, '--logdir', logdir2])
# tb2.configure(argv=[None, '--logdir', logdirNew])
tb3.configure(argv=[None, '--logdir', logdirNew_2])
# tb4.configure(argv=[None, '--logdir', logdir11])
# tb5.configure(argv=[None, '--logdir', logdir11_2])

# url = tb.launch()
# url1 = tb1.launch()
# url2 = tb2.launch()
url3 = tb3.launch()
# url4 = tb4.launch()
# url5 = tb5.launch()

# print("ML0")
# print(f"Tensorflow listening on {url}")
# print("")
# print("")
                    
# print("-------")
# print("ML2")
# print(f"Tensorflow listening on {url1}")
# print("")

# print("-------")
# print("ML_New")
# print(f"Tensorflow listening on {url2}")
# print("")

print("-------")
print("ML_New_2")
print(f"Tensorflow listening on {url3}")
# print("")

# print("-------")
# print("ML_11")
# print(f"Tensorflow listening on {url4}")
# print("")

# print("-------")
# print("ML_11_2")
# print(f"Tensorflow listening on {url5}")

time.sleep(100)